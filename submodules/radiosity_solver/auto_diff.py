"""
Auto-differentiation version of the radiosity solver.
The solver is forced to be either the hybrid solver or PR solver, and the cluster is forced to be used with 2 layers.
"""

import torch
from arguments import *
from .cluster import *
from submodules.visibility_estimator import *
from utils.sh_utils import eval_sh, eval_sh_response
from functools import partial
from tqdm import tqdm
import random as r
import _radiosity_solver_aux as _C # type: ignore

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def _estimate_one_bounce(
    handle, 
    means3D: torch.Tensor, 
    geovalues: torch.Tensor, 
    scales: torch.Tensor, 
    rots: torch.Tensor, 
    normals: torch.Tensor, 
    norm_factors: torch.Tensor, 
    emissions: torch.Tensor, 
    brdf_coeffs: torch.Tensor, 
    is_light_source: torch.Tensor, 
    active_sh_degree: int, 
    i: torch.Tensor, 
    j: torch.Tensor, 
    decay_light_source: bool = True, 
    inverse_falloff_max: float = 1., 
    min_decay: float = 1e-4
) -> torch.Tensor:
    dirs = torch.nn.functional.normalize(means3D[i] - means3D[j])
    dirs_i = (dirs[:, None, :] @ build_rotation(rots[i])).squeeze(1)
    dirs_j = (dirs[:, None, :] @ build_rotation(rots[j])).squeeze(1)
    YT_B = eval_sh(active_sh_degree, emissions[j].permute(0, 2, 1), dirs_j) # (N, 3)
    f_YT_B = torch.zeros_like(brdf_coeffs[i])
    f_YT_B[:, :(active_sh_degree + 1) ** 2, :] = eval_sh_response(active_sh_degree, YT_B, dirs_i).permute(0, 2, 1)
    f_YT_B = f_YT_B * brdf_coeffs[i] # (N, D, 3)

    non_ls_j_mask = ~is_light_source.squeeze()[j]
    # Angular factors
    V = (-normals[i] * dirs).sum(dim=-1).clamp_min(0.)
    V[non_ls_j_mask] = V[non_ls_j_mask] * (normals[j[non_ls_j_mask]] * dirs[non_ls_j_mask]).sum(dim=-1).clamp_min(0.)
    # Inverse square falloff
    if decay_light_source:
        V = V / (torch.linalg.vector_norm(means3D[i] - means3D[j], ord=2, dim=-1) ** 2 + 1E-8).clamp_min(1.)
    else:
        decay_mask = torch.logical_and(~is_light_source.squeeze()[i], ~is_light_source.squeeze()[j])
        V[decay_mask] = V[decay_mask] / (torch.linalg.vector_norm(means3D[i[decay_mask]] - means3D[j[decay_mask]], ord=2, dim=-1) ** 2 + 1E-8).clamp_min(1. / inverse_falloff_max)
    # Visibility
    # V[non_ls_j_mask] = V[non_ls_j_mask] * footprint_activation(geovalues.squeeze()[j[non_ls_j_mask]])
    V = V * estimate_visibility(handle, means3D, geovalues, scales, rots, means3D[j], means3D[i], min_decay)
    # Norm factor
    V[non_ls_j_mask] = V[non_ls_j_mask] * norm_factors.squeeze()[j[non_ls_j_mask]]

    return f_YT_B * V[:, None, None]

@torch.no_grad()
@torch.compile()
def __sample_stop_idx_s(walk_idx, num_elements, radiosity, radiosity_square, denom, valid_s, verbose):
    if walk_idx < 8:
        # First 8 Passes: Gather Information
        current_stop_idx_s = torch.arange(num_elements, device=valid_s.device, dtype=torch.int)[valid_s]
    else:
        valid_s = torch.logical_and(valid_s, denom.reshape_as(valid_s) > 1)
        if len(valid_s.reshape(-1).nonzero()) == 0:
            return None
        avg_radiosity = radiosity[valid_s] / denom[valid_s]
        var_radiosity = (radiosity_square[valid_s] + denom[valid_s] * avg_radiosity.square() - 2 * avg_radiosity * radiosity[valid_s]) / (denom[valid_s] - 1)
        sample_prob = var_radiosity.clamp_min(1e-4).sqrt().norm(dim=-1).sum(dim=-1)
        sample_prob = sample_prob.pow(0.1)
        sample_prob = sample_prob / torch.sum(sample_prob)
        if not (sample_prob > 0.).any():
            return None
        indices = torch.multinomial(sample_prob, num_samples=radiosity.size(0), replacement=True)
        current_stop_idx_s = valid_s.nonzero()[indices].int().reshape(-1)
    return current_stop_idx_s

# If you want to use the auto differentiation version, you need to
# explicitly build the BVH first and then pass the handle into this
# function. The handle needs to be valid in the both forward and 
# backward pass, i.e., created before solving and released after 
# loss backpropagation.
def solve_radiosity(
    handle, 
    means3D: torch.Tensor, 
    geovalues: torch.Tensor, 
    scales: torch.Tensor, 
    rots: torch.Tensor, 
    norm_factors: torch.Tensor, 
    emissions: torch.Tensor, 
    brdf_coeffs: torch.Tensor, 
    is_light_source: torch.Tensor, 
    active_sh_degree: int, 
    max_sh_degree: int, 
    decay_light_source: bool = True, 
    illumination_type: str = "direct", 
    num_walks: int = 128, 
    verbose: bool = False, 
    max_seed: int = 4294967295, 
    inverse_falloff_max: float = 1., 
    min_decay: float = 1e-4
):
    assert illumination_type in ["direct", "global"]
    normals = build_scaling_rotation(torch.cat([scales, torch.ones_like(scales[:, :1])], dim=-1), rots).permute(0, 2, 1)[:, -1, :]
    # handle = vis_build_bvh(means3D, scales, rots, geovalues)
    estimate_one_bounce = partial(_estimate_one_bounce, handle=handle, means3D=means3D, geovalues=geovalues, scales=scales, rots=rots, normals=normals, norm_factors=norm_factors, brdf_coeffs=brdf_coeffs, is_light_source=is_light_source, active_sh_degree=active_sh_degree, decay_light_source=decay_light_source, inverse_falloff_max=inverse_falloff_max, min_decay=min_decay)

    direct_radiosity = torch.zeros_like(emissions)

    # Direct Illumination
    emit_indices = ( is_light_source).squeeze().nonzero().squeeze().reshape(-1)
    recv_indices = (~is_light_source).squeeze().nonzero().squeeze().reshape(-1)
    len_emit_indices = len(emit_indices)
    len_recv_indices = len(recv_indices)
    emit_indices = torch.repeat_interleave(emit_indices, len_recv_indices, 0)
    recv_indices = recv_indices.repeat(len_emit_indices)
    direct_radiosity[recv_indices] = estimate_one_bounce(emissions=emissions, i=recv_indices, j=emit_indices).reshape(len_emit_indices, len_recv_indices, -1, emissions.shape[-1]).sum(0)
    if illumination_type == "direct":
        # vis_release_bvh(handle)
        return direct_radiosity
    
    # Indirect illumination
    radiosity = direct_radiosity.clone()
    radiosity_square = direct_radiosity.square().detach().requires_grad_(False)
    denom = torch.ones_like(emissions[:, :1, :1], requires_grad=False)
    valid_s = (~is_light_source).squeeze()
    num_elements = means3D.size(0)
    
    # Cluster - 2 Layers
    pl_indices = is_light_source.nonzero().reshape(-1)
    if len(pl_indices) == 0:
        cluster_info = multilayer_cluster_kernels(means3D, normals, False, 2)
    else:
        pl_indice = pl_indices[0].item()
        non_ls_cluster = multilayer_cluster_kernels(means3D[:pl_indice], normals[:pl_indice], False, 2)
        pl_cluster = multilayer_cluster_kernels(means3D[pl_indice:], normals[pl_indice:], True, 2)
        cluster_info = multilayer_merge_clusters(non_ls_cluster, pl_cluster)
    
    for walk_idx in range(num_walks) if not verbose else tqdm(range(num_walks)):
        # Next event (not differentiable)
        with torch.no_grad():
            current_stop_idx_s = __sample_stop_idx_s(walk_idx, num_elements, radiosity, radiosity_square, denom, valid_s, verbose)
            if current_stop_idx_s is None:
                break

            latest_estimate_radiosity = (radiosity / denom)
            scaled_latest_estimate_radiosity = latest_estimate_radiosity * norm_factors[:, None, :]

            # Populate `cluster_radiosities`
            for cluster_layer_idx, cluster in enumerate(cluster_info):
                if cluster_layer_idx == 0:
                    cluster["cluster_radiosities"] = sum_in_cluster(cluster, latest_estimate_radiosity)
                else:
                    cluster["cluster_radiosities"] = sum_in_cluster(cluster, cluster_info[cluster_layer_idx - 1]["cluster_radiosities"])
            
            # Sample the top layer
            next, pdf = _C.next_event_estimator(r.randint(0, max_seed), active_sh_degree, max_sh_degree, False, True, means3D, scales, rots, normals, latest_estimate_radiosity, brdf_coeffs, is_light_source, current_stop_idx_s, cluster_info[-1]["cluster_xyzs"], cluster_info[-1]["cluster_normals"], cluster_info[-1]["cluster_radiosities"], cluster_info[-1]["cluster_lss"], torch.Tensor([]).int(), decay_light_source, inverse_falloff_max, min_decay)

            valid_mask = torch.logical_and(next >= 0, ~torch.isnan(pdf))

            if not valid_mask.any().item():
                continue

            current_stop_idx_s = current_stop_idx_s[valid_mask]
            next = next[valid_mask]
            pdf = pdf[valid_mask]
            continue_flag = False

            # Sample Other Layers
            for cluster, layer_idx in reversed(list(zip(cluster_info, list(range(len(cluster_info)))))):
                sorted_next = torch.sort(next)
                next = sorted_next.values.contiguous()
                pdf = pdf[sorted_next.indices].contiguous()
                current_stop_idx_s = current_stop_idx_s[sorted_next.indices].contiguous()

                if layer_idx == 0:
                    sorted_means3D = means3D[cluster["cat_cluster_idx2idx"].long()].contiguous()
                    sorted_normals = normals[cluster["cat_cluster_idx2idx"].long()].contiguous()
                    sorted_radiosities = scaled_latest_estimate_radiosity[cluster["cat_cluster_idx2idx"].long()].contiguous()
                    sorted_is_ls = is_light_source[cluster["cat_cluster_idx2idx"].long()].contiguous()
                else:
                    sorted_means3D = cluster_info[layer_idx - 1]["cluster_xyzs"][cluster["cat_cluster_idx2idx"].long()].contiguous()
                    sorted_normals = cluster_info[layer_idx - 1]["cluster_normals"][cluster["cat_cluster_idx2idx"].long()].contiguous()
                    sorted_radiosities = cluster_info[layer_idx - 1]["cluster_radiosities"][cluster["cat_cluster_idx2idx"].long()].contiguous()
                    sorted_is_ls = cluster_info[layer_idx - 1]["cluster_lss"][cluster["cat_cluster_idx2idx"].long()].contiguous()
                
                next, pdf = _C.in_cluster_next_event_estimator(r.randint(0, max_seed), active_sh_degree, max_sh_degree, False, True, means3D, normals, brdf_coeffs, is_light_source, current_stop_idx_s, next, pdf, cluster["cluster_end_offset"], cluster["cat_cluster_idx2idx"], cluster["cat_cluster_idx"], sorted_means3D, sorted_normals, sorted_radiosities, sorted_is_ls, torch.Tensor([]).int(), decay_light_source, inverse_falloff_max, min_decay)

                valid_mask = torch.logical_and(next >= 0, ~torch.isnan(pdf))

                if not valid_mask.any().item():
                    continue_flag = True
                    break

                current_stop_idx_s = current_stop_idx_s[valid_mask]
                next = next[valid_mask]
                pdf = pdf[valid_mask]
            
            if continue_flag:
                continue
        
        x = direct_radiosity[current_stop_idx_s] + torch.reciprocal(pdf + 1E-8)[:, None, None] * estimate_one_bounce(emissions=radiosity / denom.clone().detach(), i=current_stop_idx_s, j=next)
        radiosity.scatter_add_(0, current_stop_idx_s.long()[:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x)
        with torch.no_grad():
            radiosity_square.scatter_add_(0, current_stop_idx_s.long()[:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x.square())
            denom.scatter_add_(0, current_stop_idx_s.long()[:, None, None], torch.ones_like(current_stop_idx_s[:, None, None], dtype=denom.dtype))
    
    # vis_release_bvh(handle)
    # Can directly use the indirect radiosity here
    # as the direct illumination is set to run for only 1 step.
    # So, the only difference will be there is no emission for light sources, 
    # while there should. But since we do not render light sources anyway, 
    # no need to account for that.
    return radiosity / denom.detach()