#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import *
import sys
from scene import *
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import *
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import json
from time import strftime, gmtime, time

def linear_schedule_fn(init_value, final_value, total_steps, start_steps=0):
    def schedule(step):
        step = min(max(step - start_steps, 0), total_steps - start_steps)
        return init_value + (final_value - init_value) * step / (total_steps - start_steps)
    return schedule

def training(dataset, opt, pipe, test_interval, saving_iterations, checkpoint_iterations, checkpoint, quiet):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, quiet=quiet)
    gaussians = GaussianModel(dataset)
    light_sources = LightModel(dataset)
    scene = Scene(dataset, gaussians, light_sources, geovalue_init=opt.geovalue_init, envmap_init=opt.init_env_map)
    gaussians.training_setup(opt)
    light_sources.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        light_sources.load(checkpoint.replace("chkpnt", "ls_chkpnt"))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_idx_stack = None
    ema_elapse_for_log = 0.0

    start_time = time()
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    gaussians.active_sh_degree = opt.init_sh_degree

    try:
        for iteration in range(first_iter, opt.iterations + 1):
            gaussians.update_learning_rate(iteration)
            num_walks = int(linear_schedule_fn(opt.mc_init_walks, opt.mc_final_walks, opt.mc_decay_steps, opt.mc_start_steps)(iteration)) if pipe.solver_type != "PR" else 0
            # Overcome some weird numerical instability problem
            # - 1e-4 used to work across all cases, but in some latest tests, it does not always work.
            # - Higher min decay generally leads to smoother geometry, while lower min decay leads to better details.
            # We therefore use a linear scheduling to have both smooth geometry and details.
            min_decay = linear_schedule_fn(opt.min_decay_init, opt.min_decay_final, opt.min_decay_decay_steps, opt.min_decay_start_steps)(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if min((iteration // 1000) * (2 if gaussians.max_sh_degree > 4 else 1) + opt.init_sh_degree, dataset.sh_degree) > gaussians.active_sh_degree:
                if gaussians.max_sh_degree > 4:
                    gaussians.twoupSHdegree()
                else:
                    gaussians.oneupSHdegree()

            iter_start.record()

            # Pick a random Camera
            if not viewpoint_idx_stack:
                viewpoint_idx_stack = list(range(len(scene.getTrainCameras())))

            viewpoint_cam = scene.getTrainCameras()[viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack)-1))]

            render_pkg = renderGI(viewpoint_cam, gaussians, light_sources.from_camera_if_possible(viewpoint_cam), pipe, background, override_solver_settings={"inverse_falloff_max": dataset.max_inverse_falloff, "num_walks": num_walks, "min_decay": min_decay, "gradient_num_walks": num_walks, "use_cluster": not pipe.not_use_cluster})
            
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if viewpoint_cam.exr_image is None:
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam.exr_image.cuda()
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # regularization
            lambda_normal = opt.lambda_normal if iteration > opt.ndc_start_iteration else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
            lambda_alpha = opt.lambda_alpha
            lambda_env_map = opt.lambda_env_map if iteration < opt.densify_clone_only_from_iter else 0.0
            percent_dense = opt.percent_dense if iteration > opt.densify_absgrad_start_iteration else 0.01

            rend_dist = render_pkg["rend_dist"] * gt_alpha_mask
            rend_normal = render_pkg['rend_normal'] * gt_alpha_mask
            surf_normal = render_pkg['surf_normal'] * gt_alpha_mask
            rend_alpha = render_pkg['rend_alpha']
            
            if iteration <= 7000:
                surf_normal = surf_normal.detach()

            normal_loss = lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0)).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
            alpha_loss = lambda_alpha * (rend_alpha - gt_alpha_mask).abs().mean()
            env_map_loss = lambda_env_map * light_sources.get_emissions.norm(dim=-1)[:, 0].mean()
            
            # loss
            total_loss = loss + dist_loss + normal_loss + alpha_loss + env_map_loss

            denom_filter = visibility_filter
            visibility_filter = torch.logical_and(visibility_filter, ((gaussians.get_xyz - viewpoint_cam.camera_center) * gaussians.get_normal).sum(dim=-1) < 0)

            total_loss.backward()

            iter_end.record()
            torch.cuda.synchronize()

            with torch.no_grad():
                # Progress bar
                ema_elapse_for_log = 0.4 * iter_start.elapsed_time(iter_end) + 0.6 * ema_elapse_for_log

                if iteration % 10 == 0 and iteration > 0:
                    loss_dict = {
                        "Points": f"{len(gaussians.get_xyz)}", 
                        "Walks": f"{num_walks}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                training_report(tb_writer, iteration, {
                    "reg_loss": Ll1, 
                    "total_loss": loss, 
                    "dist_loss": dist_loss, 
                    "normal_loss": normal_loss, 
                    "alpha_loss": alpha_loss, 
                    "env_map_loss": env_map_loss, 
                }, l1_loss, ema_elapse_for_log, test_interval, scene, renderGI, (pipe, background), override_solver_settings={"inverse_falloff_max": dataset.max_inverse_falloff, "num_walks": num_walks, "min_decay": min_decay, "use_cluster": not pipe.not_use_cluster})

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                
                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.maximum(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, denom_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_absgrad_threshold, opt.geovalue_cull, scene.cameras_extent, iteration > 3000, percent_dense=percent_dense, use_absgrad=iteration >= opt.densify_absgrad_start_iteration, densify_type='clone-only' if iteration > opt.densify_clone_only_from_iter else 'densify-clone')
                
                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    light_sources.save(scene.model_path + "/ls_chkpnt" + str(iteration) + ".pth")

                gaussians.optimizer.step()
                light_sources.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                light_sources.optimizer.zero_grad(set_to_none = True)
    
    except KeyboardInterrupt:
        print("Aborted!")
        sys.exit(0)
    
    with open(os.path.join(dataset.model_path, "training_time.json"), 'w') as f:
        json.dump({ "start_time": start_time, "stop_time": time() }, f)

def prepare_output_and_logger(args, opt, quiet=False):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = os.path.basename(args.source_path) + '-' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        args.model_path = os.path.join(f"./output/", unique_str)
    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    with open(os.path.join(args.model_path, "training_options.json"), 'w') as f:
        json.dump(vars(opt), f)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and not quiet:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, loss_dict, l1_loss, elapsed, test_interval, scene : Scene, renderFunc, renderArgs, override_solver_settings={}):
    if tb_writer:
        for name in loss_dict:
            tb_writer.add_scalar(f'train_loss_patches/{name}', loss_dict[name].item(), iteration)
        tb_writer.add_scalar('scene/iter_time', elapsed, iteration)
        tb_writer.add_scalar('scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration > 0 and iteration % test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()[:5]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                use_env_map = False
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, scene.light_sources.from_camera_if_possible(viewpoint), *renderArgs, override_solver_settings=override_solver_settings)
                    use_env_map = use_env_map or scene.light_sources.from_camera_if_possible(viewpoint).is_directional_light

                    if viewpoint.exr_image is None:
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.exr_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        rend_alpha = render_pkg['rend_alpha']
                        rend_normal = -(render_pkg["rend_normal"].permute(1, 2, 0) @ viewpoint.view_world_transform[:3, :3].T).permute(2, 0, 1) * 0.5 + 0.5
                        surf_normal = -(render_pkg["surf_normal"].permute(1, 2, 0) @ viewpoint.view_world_transform[:3, :3].T).permute(2, 0, 1) * 0.5 + 0.5
                        tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                        rend_dist = render_pkg["rend_dist"]
                        rend_dist = colormap(rend_dist.cpu().numpy()[0])
                        tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)

                        if iteration == test_interval:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    if use_env_map:
                        tb_writer.add_images("env_map", torch.nn.functional.interpolate(scene.light_sources.get_env_map[None], scale_factor=4, mode='bilinear', align_corners=False, antialias=True).cpu().numpy(), global_step=iteration)
                    
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_histogram("scene/geovalue_histogram", torch.nan_to_num(scene.gaussians.get_geovalue), iteration)
                    tb_writer.add_histogram("scene/blending_histogram", torch.nan_to_num(scene.gaussians.get_blending), iteration)
                    tb_writer.add_histogram("scene/diffuse_albedo_histogram", torch.nan_to_num(scene.gaussians.get_real_diffuse_albedos), iteration)
                    tb_writer.add_histogram("scene/specular_albedo_histogram", torch.nan_to_num(scene.gaussians.get_real_specular_albedos), iteration)
                    tb_writer.add_histogram("scene/shininess_histogram", torch.nan_to_num(scene.gaussians.get_shininess), iteration)
                    tb_writer.add_histogram("scene/scaling_histogram", torch.nan_to_num(scene.gaussians.get_scaling), iteration)
                    tb_writer.add_histogram("scene/norm_factor_histogram", torch.nan_to_num(scene.gaussians.get_real_norm_factor), iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_interval, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.quiet)

    # All done
    print("\nTraining complete.")