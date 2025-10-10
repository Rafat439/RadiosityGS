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
import numpy as np
from torch import nn
from arguments import *
from functools import partial
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud, estimateTangentPlane
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling_rotation

from simple_knn._C import distCUDA2 # type: ignore
from submodules.radiosity_solver.util import non_ls_lambda_activation

def construct_phong_brdf_coeffs(
    diffuse_albedo, 
    specular_albedo, 
    shininess, 
    active_sh_degree, 
    max_sh_degree
):
    '''Exact Phong BRDF. Not a complex one.'''
    coeffs = []
    # From Ramamoorthi's 'A Signal-Processing Framework for Inverse Rendering' Appendix
    zeros = torch.ones_like(specular_albedo) * 1E-8
    
    coeffs.append(4 * diffuse_albedo + specular_albedo)
    if max_sh_degree > 0:
        c1 = (specular_albedo * (shininess + 1) / (shininess + 2)) if active_sh_degree >= 1 else zeros
        coeffs = coeffs + [ -c1, c1, -c1 ]
        if max_sh_degree > 1:
            c2 = (specular_albedo * (shininess) / (shininess + 3)) if active_sh_degree >= 2 else zeros
            coeffs = coeffs + [ c2, -c2, c2, -c2, c2 ]
            if max_sh_degree > 2:
                c3 = (c1 * (shininess - 1) / (shininess + 4)) if active_sh_degree >= 3 else zeros
                coeffs = coeffs + [ -c3, c3, -c3, c3, -c3, c3, -c3 ]
                if max_sh_degree > 3:
                    c4 = (c2 * (shininess - 2) / (shininess + 5)) if active_sh_degree >= 4 else zeros
                    coeffs = coeffs + [ c4, -c4, c4, -c4, c4, -c4, c4, -c4, c4 ]
                    if max_sh_degree > 4:
                        c5 = (c3 * (shininess - 3) / (shininess + 6)) if active_sh_degree >= 5 else zeros
                        coeffs = coeffs + [ -c5, c5, -c5, c5, -c5, c5, -c5, c5, -c5, c5, -c5 ]
                        if max_sh_degree > 5:
                            c6 = (c4 * (shininess - 4) / (shininess + 7)) if active_sh_degree >= 6 else zeros
                            coeffs = coeffs + [ c6, -c6, c6, -c6, c6, -c6, c6, -c6, c6, -c6, c6, -c6, c6 ]
                            if max_sh_degree > 6:
                                c7 = (c5 * (shininess - 5) / (shininess + 8)) if active_sh_degree >= 7 else zeros
                                coeffs = coeffs + [ -c7, c7, -c7, c7, -c7, c7, -c7, c7, -c7, c7, -c7, c7, -c7, c7, -c7 ]
                                if max_sh_degree > 7:
                                    c8 = (c6 * (shininess - 6) / (shininess + 9)) if active_sh_degree >= 8 else zeros
                                    coeffs = coeffs + [ c8, -c8, c8, -c8, c8, -c8, c8, -c8, c8, -c8, c8, -c8, c8, -c8, c8, -c8, c8 ]
                                    if max_sh_degree > 8:
                                        c9 = (c7 * (shininess - 7) / (shininess + 10)) if active_sh_degree >= 9 else zeros
                                        coeffs = coeffs + [ -c9, c9, -c9, c9, -c9, c9, -c9, c9, -c9, c9, -c9, c9, -c9, c9, -c9, c9, -c9, c9, -c9 ]
    return torch.stack(coeffs, dim=1).contiguous() # (N, (deg + 1)^2, 3)

def softexp(x: torch.Tensor, bound: float = 1000.):
    y = torch.where(x < np.log(bound), torch.exp(x), bound * (x - np.log(bound)) + bound)
    return y

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        if self.max_scaling > 0:
            self.scaling_activation = lambda x: self.max_scaling * torch.sigmoid(x)
            self.inverse_scaling_activation = lambda y: inverse_sigmoid(torch.clamp(y / self.max_scaling, 1e-8, 1. - 1e-8))
        else:
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.geovalue_activation = lambda x: self.geovalue_mul * torch.sigmoid(x)
        self.inverse_geovalue_activation = lambda y: inverse_sigmoid(y / self.geovalue_mul)
        self.rotation_activation = torch.nn.functional.normalize

        self.shininess_activation = lambda x: torch.sigmoid(x) * self.max_shininess
        self.inverse_shininess_activation = lambda y: inverse_sigmoid(y / self.max_shininess)

        self.albedo_activation = lambda x: torch.sigmoid(x)
        self.inverse_albedo_activation = lambda y: inverse_sigmoid(y)
        self.lambda_activation = partial(softexp, bound=self.max_lambda)

    def __init__(self, dataset):
        self.active_sh_degree = dataset.sh_degree
        self.max_sh_degree = dataset.sh_degree
        self.max_shininess = ((dataset.sh_degree + 1) ** 2) / 2.5 # We double in practice
        self.max_scaling = dataset.max_scaling
        self.max_lambda = dataset.max_lambda
        self.geovalue_mul = dataset.geovalue_mul

        self._xyz = None
        self._scaling = None
        self._rotation = None
        self._geovalue = None
        self._norm_factor = None
        self._blending = None
        self._shininess = None
        self._diffuse_albedos = None
        self._specular_albedos = None

        
        self.max_radii2D = None
        self.xyz_gradient_accum = None
        self.xyz_absgradient_accum = None
        self.denom = None
        self.optimizer = None

        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._geovalue,
            self._norm_factor, 
            self._blending, 
            self._shininess, 
            self._diffuse_albedos, 
            self._specular_albedos, 
            
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_absgradient_accum, 
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._scaling, 
        self._rotation, 
        self._geovalue,
        self._norm_factor, 
        self._blending, 
        self._shininess, 
        self._diffuse_albedos, 
        self._specular_albedos, 
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_absgradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_absgradient_accum = xyz_absgradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_normal(self):
        scales = self.get_scaling
        rots = self.get_rotation
        normals = build_scaling_rotation(torch.cat([scales, torch.ones_like(scales[:, :1])], dim=-1), rots).permute(0, 2, 1)[:, -1, :]
        return normals
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_blending(self):
        return torch.sigmoid(self._blending)
    
    @property
    def get_real_diffuse_albedos(self):
        return self.albedo_activation(self._diffuse_albedos)
    
    @property
    def get_real_specular_albedos(self):
        return self.albedo_activation(self._specular_albedos)
    
    @property
    def get_shininess(self):
        return self.shininess_activation(self._shininess)
    
    @property
    def get_diffuse_albedos(self):
        return self.get_real_diffuse_albedos * self.get_blending
    
    @property
    def get_specular_albedos(self):
        return self.get_real_specular_albedos * (1 - self.get_blending)
    
    @property
    def get_geovalue(self):
        return self.geovalue_activation(self._geovalue)

    @property
    def get_brdf_coeffs(self):
        # The alpha is fused into the albedo as an integrated optimized component.
        # footprint_activation(self.get_geovalue)[:, :, None]
        return construct_phong_brdf_coeffs(self.get_diffuse_albedos, self.get_specular_albedos, self.get_shininess, self.active_sh_degree, self.max_sh_degree)
    
    @property
    def get_real_norm_factor(self):
        return self.lambda_activation(self._norm_factor)

    @property
    def get_norm_factor(self):
        return self.get_real_norm_factor * non_ls_lambda_activation(self.get_geovalue, self.get_scaling)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def twoupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 2

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_geovalue : float):
        self.init_geovalue = init_geovalue
        self.spatial_lr_scale = 5. # spatial_lr_scale
        print("Max scaling : ", self.max_scaling)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = self.inverse_scaling_activation(torch.sqrt(dist2))[...,None].repeat(1, 2)
        # rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots = estimateTangentPlane(pcd.points)

        shininess = self.inverse_shininess_activation((self.max_shininess / 2) * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        diffuse_albedo = self.inverse_albedo_activation(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        specular_albedo = self.inverse_albedo_activation(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        blending = torch.tensor(np.asarray(pcd.colors) * 0 + 0.0).float().cuda()

        geovalue = self.inverse_geovalue_activation(init_geovalue * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        norm_factor = (0. * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._shininess = nn.Parameter(shininess.requires_grad_(True))
        self._diffuse_albedos = nn.Parameter(diffuse_albedo.requires_grad_(True))
        self._specular_albedos = nn.Parameter(specular_albedo.requires_grad_(True))
        self._blending = nn.Parameter(blending.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._geovalue = nn.Parameter(geovalue.requires_grad_(True))
        self._norm_factor = nn.Parameter(norm_factor.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_absgradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._shininess], 'lr': training_args.shininess_lr, "name": "shininess"},
            {'params': [self._diffuse_albedos], 'lr': training_args.diffuse_albedo_lr, "name": "diffuse_albedo"},
            {'params': [self._specular_albedos], 'lr': training_args.specular_albedo_lr, "name": "specular_albedo"},
            {'params': [self._geovalue], 'lr': training_args.geovalue_lr_init, "name": "geovalue"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}, 
            {'params': [self._blending], 'lr': training_args.blending_lr, "name": "blending"}, 
            {'params': [self._norm_factor], 'lr': training_args.norm_factor_lr, "name": "norm_factor"}
        ]
        # For slightly smoother gradients due to the variance
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.geovalue_scheduler_args = get_expon_lr_func(lr_init=training_args.geovalue_lr_init, 
                                                         lr_final=training_args.geovalue_lr_final, 
                                                         max_steps=training_args.geovalue_lr_max_steps)
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "geovalue":
                lr = self.geovalue_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._shininess.shape[1]):
            l.append('s_{}'.format(i))
        for i in range(self._blending.shape[1]):
            l.append('b_{}'.format(i))
        for i in range(self._diffuse_albedos.shape[1]):
            l.append('ad_{}'.format(i))
        for i in range(self._specular_albedos.shape[1]):
            l.append('as_{}'.format(i))
        l.append('geovalue')
        l.append('norm_factor')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        shininess = self._shininess.detach().cpu().numpy()
        blending = self._blending.detach().cpu().numpy()
        diffuse_albedo = self._diffuse_albedos.detach().cpu().numpy()
        specular_albedo = self._specular_albedos.detach().cpu().numpy()
        geovalue = self._geovalue.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        norm_factor = self._norm_factor.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, shininess, blending, diffuse_albedo, specular_albedo, geovalue, norm_factor, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        geovalue = np.asarray(plydata.elements[0]["geovalue"])[..., np.newaxis]
        norm_factor = np.asarray(plydata.elements[0]["norm_factor"])[..., np.newaxis]
        
        shininess_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("s_")]
        shininess_names = sorted(shininess_names, key = lambda x: int(x.split('_')[-1]))
        shininess = np.zeros((xyz.shape[0], len(shininess_names)))
        for idx, attr_name in enumerate(shininess_names):
            shininess[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        blending_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("b_")]
        blending_names = sorted(blending_names, key = lambda x: int(x.split('_')[-1]))
        blending = np.zeros((xyz.shape[0], len(blending_names)))
        for idx, attr_name in enumerate(blending_names):
            blending[:, idx] = np.asarray(plydata.elements[0][attr_name])

        diffuse_albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ad_")]
        diffuse_albedo_names = sorted(diffuse_albedo_names, key = lambda x: int(x.split('_')[-1]))
        diffuse_albedo = np.zeros((xyz.shape[0], len(diffuse_albedo_names)))
        for idx, attr_name in enumerate(diffuse_albedo_names):
            diffuse_albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

        specular_albedo_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("as_")]
        specular_albedo_names = sorted(specular_albedo_names, key = lambda x: int(x.split('_')[-1]))
        specular_albedo = np.zeros((xyz.shape[0], len(specular_albedo_names)))
        for idx, attr_name in enumerate(specular_albedo_names):
            specular_albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shininess = nn.Parameter(torch.tensor(shininess, dtype=torch.float, device="cuda").requires_grad_(True))
        self._blending = nn.Parameter(torch.tensor(blending, dtype=torch.float, device="cuda").requires_grad_(True))
        self._diffuse_albedos = nn.Parameter(torch.tensor(diffuse_albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._specular_albedos = nn.Parameter(torch.tensor(specular_albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._geovalue = nn.Parameter(torch.tensor(geovalue, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._norm_factor = nn.Parameter(torch.tensor(norm_factor, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        if self.optimizer is None:
            return { name: tensor }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "network":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask].contiguous()
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask].contiguous()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].contiguous().requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].contiguous().requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def __apply_mask(self, valid_points_mask):
        if self.optimizer is not None:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._blending = optimizable_tensors["blending"]
            self._shininess = optimizable_tensors["shininess"]
            self._diffuse_albedos = optimizable_tensors["diffuse_albedo"]
            self._specular_albedos = optimizable_tensors["specular_albedo"]
            self._geovalue = optimizable_tensors["geovalue"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._norm_factor = optimizable_tensors["norm_factor"]
        else:
            self._xyz = torch.nn.Parameter(self._xyz[valid_points_mask].contiguous().requires_grad_(True))
            self._blending = torch.nn.Parameter(self._blending[valid_points_mask].contiguous().requires_grad_(True))
            self._shininess = torch.nn.Parameter(self._shininess[valid_points_mask].contiguous().requires_grad_(True))
            self._diffuse_albedos = torch.nn.Parameter(self._diffuse_albedos[valid_points_mask].contiguous().requires_grad_(True))
            self._specular_albedos = torch.nn.Parameter(self._specular_albedos[valid_points_mask].contiguous().requires_grad_(True))
            self._geovalue = torch.nn.Parameter(self._geovalue[valid_points_mask].contiguous().requires_grad_(True))
            self._scaling = torch.nn.Parameter(self._scaling[valid_points_mask].contiguous().requires_grad_(True))
            self._rotation = torch.nn.Parameter(self._rotation[valid_points_mask].contiguous().requires_grad_(True))
            self._norm_factor = torch.nn.Parameter(self._norm_factor[valid_points_mask].contiguous().requires_grad_(True))

        if self.xyz_gradient_accum is not None: self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask].contiguous()
        if self.xyz_absgradient_accum is not None: self.xyz_absgradient_accum = self.xyz_absgradient_accum[valid_points_mask].contiguous()
        if self.denom is not None: self.denom = self.denom[valid_points_mask].contiguous()
        
        if self.max_radii2D is not None: self.max_radii2D = self.max_radii2D[valid_points_mask].contiguous()
        self.adjacent_matrix = None
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        return self.__apply_mask(valid_points_mask)
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        if self.optimizer is None:
            return {"xyz": torch.cat((self._xyz, tensors_dict["xyz"])),
                "blending": torch.cat((self._blending, tensors_dict["blending"])), 
                "shininess": torch.cat((self._shininess, tensors_dict["shininess"])), 
                "diffuse_albedo": torch.cat((self._diffuse_albedos, tensors_dict["diffuse_albedo"])),
                "specular_albedo": torch.cat((self._specular_albedos, tensors_dict["specular_albedo"])),
                "geovalue": torch.cat((self._geovalue, tensors_dict["geovalue"])),
                "scaling" : torch.cat((self._scaling, tensors_dict["scaling"])),
                "rotation" : torch.cat((self._rotation, tensors_dict["rotation"])), 
                "norm_factor": torch.cat((self._norm_factor, tensors_dict["norm_factor"]))}
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group["name"] in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_blending, new_shininess, new_diffuse_albedo, new_specular_albedo, new_geovalue, new_scaling, new_rotation, new_norm_factor):
        d = {"xyz": new_xyz,
        "blending": new_blending, 
        "shininess": new_shininess, 
        "diffuse_albedo": new_diffuse_albedo,
        "specular_albedo": new_specular_albedo,
        "geovalue": new_geovalue,
        "scaling" : new_scaling,
        "rotation" : new_rotation, 
        "norm_factor": new_norm_factor}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._blending = optimizable_tensors["blending"]
        self._shininess = optimizable_tensors["shininess"]
        self._diffuse_albedos = optimizable_tensors["diffuse_albedo"]
        self._specular_albedos = optimizable_tensors["specular_albedo"]
        self._geovalue = optimizable_tensors["geovalue"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._norm_factor = optimizable_tensors["norm_factor"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_absgradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent, percent_dense):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_blending = self._blending[selected_pts_mask]
        new_shininess = self._shininess[selected_pts_mask]
        new_diffuse_albedo = self._diffuse_albedos[selected_pts_mask]
        new_specular_albedo = self._specular_albedos[selected_pts_mask]
        # new_geovalue = self._geovalue[selected_pts_mask]
        new_geovalue = self.inverse_geovalue_activation(torch.ones_like(self._geovalue[selected_pts_mask]) * self.init_geovalue)
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_norm_factor = self._norm_factor[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_blending, new_shininess, new_diffuse_albedo, new_specular_albedo, new_geovalue, new_scaling, new_rotation, new_norm_factor)
    
    def densify_and_evsplit(self, grads, grad_threshold, scene_extent, percent_dense, N=4, delta_dist=0.25):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask]
        stds = torch.cat([stds, 0. * torch.ones_like(stds[:,:1])], dim=-1)
        rots = build_rotation(self._rotation[selected_pts_mask])
        if N == 2:
            # Split along the shortest axis
            n = torch.where((self.get_scaling[:, 0] > self.get_scaling[:, 1])[selected_pts_mask, None].repeat(1, 3), torch.tensor([1., 0., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1), torch.tensor([0., 1., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1))
            tau = torch.sum(stds * n, dim=-1, keepdim=True)
            L = stds.square() * n
            delta_xyz = np.sqrt(delta_dist) * L / tau
            new_xyz = torch.cat((
                self.get_xyz[selected_pts_mask] - (rots @ delta_xyz[:, :, None]).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ delta_xyz[:, :, None]).squeeze(-1)
            ))
            new_scaling = (self.get_scaling[selected_pts_mask].square() - (delta_dist) * L[:, :2]).sqrt()
            new_scaling = torch.where(new_scaling > 1e-4, new_scaling, self.get_scaling[selected_pts_mask])
            new_scaling = self.inverse_scaling_activation(new_scaling).repeat(N,1)
        elif N == 4:
            # Split along both axes
            n0 = torch.tensor([1., 0., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1)
            n1 = torch.tensor([0., 1., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1)
            tau0 = torch.sum(stds * n0, dim=-1, keepdim=True)
            tau1 = torch.sum(stds * n1, dim=-1, keepdim=True)
            L0 = stds.square() * n0
            L1 = stds.square() * n1
            delta_xyz0 = np.sqrt(delta_dist) * L0 / tau0
            delta_xyz1 = np.sqrt(delta_dist) * L1 / tau1
            new_xyz = torch.cat((
                self.get_xyz[selected_pts_mask] + (rots @ (- delta_xyz0[:, :, None] - delta_xyz1[:, :, None])).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ (- delta_xyz0[:, :, None] + delta_xyz1[:, :, None])).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ (+ delta_xyz0[:, :, None] - delta_xyz1[:, :, None])).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ (+ delta_xyz0[:, :, None] + delta_xyz1[:, :, None])).squeeze(-1)
            ))
            new_scaling = (self.get_scaling[selected_pts_mask].square() - (delta_dist) * L0[:, :2] - (delta_dist) * L1[:, :2]).sqrt()
            new_scaling = torch.where(new_scaling > 1e-4, new_scaling, self.get_scaling[selected_pts_mask])
            new_scaling = self.inverse_scaling_activation(new_scaling).repeat(N,1)

        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_blending = self._blending[selected_pts_mask].repeat(N,1)
        new_shininess = self._shininess[selected_pts_mask].repeat(N,1)
        # new_geovalue = self._geovalue[selected_pts_mask].repeat(N,1)
        # new_geovalue = self.inverse_geovalue_activation(torch.maximum(inverse_S_activation(S_activation(self.get_geovalue[selected_pts_mask]) / N), self.init_geovalue * torch.ones_like(self._geovalue[selected_pts_mask]))).repeat(N,1)
        new_geovalue = self.inverse_geovalue_activation(inverse_S_activation(S_activation(self.get_geovalue[selected_pts_mask]) / N)).repeat(N,1)
        new_diffuse_albedo = self._diffuse_albedos[selected_pts_mask].repeat(N,1)
        new_specular_albedo = self._specular_albedos[selected_pts_mask].repeat(N,1)
        new_norm_factor = self._norm_factor[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_blending, new_shininess, new_diffuse_albedo, new_specular_albedo, new_geovalue, new_scaling, new_rotation, new_norm_factor)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_by_evclone(self, grads, grad_threshold, N=4, delta_dist=0.25):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold

        stds = self.get_scaling[selected_pts_mask]
        stds = torch.cat([stds, 0. * torch.ones_like(stds[:,:1])], dim=-1)
        rots = build_rotation(self._rotation[selected_pts_mask])
        if N == 2:
            # Split along the shortest axis
            n = torch.where((self.get_scaling[:, 0] > self.get_scaling[:, 1])[selected_pts_mask, None].repeat(1, 3), torch.tensor([1., 0., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1), torch.tensor([0., 1., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1))
            tau = torch.sum(stds * n, dim=-1, keepdim=True)
            L = stds.square() * n
            delta_xyz = np.sqrt(delta_dist) * L / tau
            new_xyz = torch.cat((
                self.get_xyz[selected_pts_mask] - (rots @ delta_xyz[:, :, None]).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ delta_xyz[:, :, None]).squeeze(-1)
            ))
            new_scaling = (self.get_scaling[selected_pts_mask].square() - (delta_dist) * L[:, :2]).sqrt()
            new_scaling = torch.where(new_scaling > 1e-4, new_scaling, self.get_scaling[selected_pts_mask])
            new_scaling = self.inverse_scaling_activation(new_scaling).repeat(N,1)
        elif N == 4:
            # Split along both axes
            n0 = torch.tensor([1., 0., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1)
            n1 = torch.tensor([0., 1., 0.], device="cuda")[None, :].repeat(selected_pts_mask.sum().item(), 1)
            tau0 = torch.sum(stds * n0, dim=-1, keepdim=True)
            tau1 = torch.sum(stds * n1, dim=-1, keepdim=True)
            L0 = stds.square() * n0
            L1 = stds.square() * n1
            delta_xyz0 = np.sqrt(delta_dist) * L0 / tau0
            delta_xyz1 = np.sqrt(delta_dist) * L1 / tau1
            new_xyz = torch.cat((
                self.get_xyz[selected_pts_mask] + (rots @ (- delta_xyz0[:, :, None] - delta_xyz1[:, :, None])).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ (- delta_xyz0[:, :, None] + delta_xyz1[:, :, None])).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ (+ delta_xyz0[:, :, None] - delta_xyz1[:, :, None])).squeeze(-1), 
                self.get_xyz[selected_pts_mask] + (rots @ (+ delta_xyz0[:, :, None] + delta_xyz1[:, :, None])).squeeze(-1)
            ))
            new_scaling = (self.get_scaling[selected_pts_mask].square() - (delta_dist) * L0[:, :2] - (delta_dist) * L1[:, :2]).sqrt()
            new_scaling = torch.where(new_scaling > 1e-4, new_scaling, self.get_scaling[selected_pts_mask])
            new_scaling = self.inverse_scaling_activation(new_scaling).repeat(N,1)

        new_geovalue = self.inverse_geovalue_activation(torch.ones_like(self._geovalue[selected_pts_mask]) * self.init_geovalue).repeat(N,1)
        # new_geovalue = self.inverse_geovalue_activation(inverse_S_activation(S_activation(self.get_geovalue[selected_pts_mask]) / N)).repeat(N,1)
        # new_geovalue = self._geovalue[selected_pts_mask].repeat(N,1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_blending = self._blending[selected_pts_mask].repeat(N,1)
        new_shininess = self._shininess[selected_pts_mask].repeat(N,1)
        new_diffuse_albedo = self._diffuse_albedos[selected_pts_mask].repeat(N,1)
        new_specular_albedo = self._specular_albedos[selected_pts_mask].repeat(N,1)
        new_norm_factor = self._norm_factor[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_blending, new_shininess, new_diffuse_albedo, new_specular_albedo, new_geovalue, new_scaling, new_rotation, new_norm_factor)

    def prune(self, min_geovalue, extent = None, max_screen_size = None):
        prune_mask = (self.get_geovalue < min_geovalue).squeeze()
        if max_screen_size:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            small_points_ws = self.get_scaling.max(dim=1).values < 1e-3
            # print(f"Prune {prune_mask.sum().item()} too transparent, {small_points_ws.sum().item()} too small, and {big_points_ws.sum().item()} too big.")
            prune_mask |= torch.logical_or(big_points_ws, small_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def densify_and_prune(self, max_grad, max_absgrad, min_geovalue, extent, max_screen_size, percent_dense, use_absgrad=True, densify_type=None):
        self.prune(min_geovalue, extent, max_screen_size)

        if densify_type == 'clone-only':
            grads = self.xyz_gradient_accum / self.denom
            absgrads = self.xyz_absgradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            absgrads[absgrads.isnan()] = 0.0

            if use_absgrad:
                self.densify_by_evclone(absgrads, max_absgrad)
            else:
                self.densify_by_evclone(grads, max_grad)
        else:
            grads = self.xyz_gradient_accum / self.denom
            absgrads = self.xyz_absgradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            absgrads[absgrads.isnan()] = 0.0
            
            if use_absgrad:
                self.densify_and_clone(grads, max_grad, extent, percent_dense)
                self.densify_and_evsplit(absgrads, max_absgrad, extent, percent_dense)
            else:
                self.densify_and_clone(grads, max_grad, extent, percent_dense)
                self.densify_and_evsplit(grads, max_grad, extent, percent_dense)

    def add_densification_stats(self, viewspace_point_tensor, update_filter, denom_update_filter):
        assert viewspace_point_tensor.grad.shape[-1] == 4
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.xyz_absgradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[denom_update_filter] += 1