import torch
import random as r
from time import time
from tqdm import tqdm
from typing import Optional, Tuple, Union, Literal
from submodules.one_bounce_estimator import OneBounceEstimator
from .util import *
from .cluster import *
import _radiosity_solver_aux as _C # type: ignore

__all__ = [
    'solve_radiosity_by_monte_carlo'
]

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
        # Remaining Passes: Allocate More Resources to High-variance Regions
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

@torch.no_grad()
def solve_radiosity_by_monte_carlo(
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
    max_sh_degree: int, 
    use_brdf_importance_sampling: bool, 
    num_walks: int = 1, 
    channel_clamp_type: Union[Literal['luminance'], Literal['norm']] = 'luminance', 
    max_seed: int = 4294967295, 
    verbose: bool = False, 
    use_cluster: bool = False, 
    nlayer_cluster: int = 2, 
    estimator: OneBounceEstimator = None, 
    cluster_info: Tuple[torch.Tensor, torch.Tensor] = None, 
    form_factor_cache: Optional[torch.Tensor] = [], 
    packed_emitter_receiver: torch.Tensor = None, 
    decay_light_source: bool = True, 
    inverse_falloff_max: float = 1., 
    min_decay: float = 1e-4
) -> torch.Tensor:
    estimator = OneBounceEstimator(means3D, geovalues, scales, rots, normals, norm_factors, active_sh_degree, max_sh_degree, brdf_coeffs, max_seed) if estimator is None else estimator
    form_factor_cache = form_factor_cache if form_factor_cache is not None else torch.Tensor([]).int()
    num_elements = means3D.size(0)

    radiosity = emissions.clone()
    radiosity_square = radiosity.square()
    denom = torch.ones_like(emissions[:, :1, :1])
    valid_s = (~is_light_source).squeeze()
    
    # Cluster Once
    if use_cluster:
        if cluster_info is None:
            # Assume `is_light_source` is sorted.
            pl_indices = is_light_source.nonzero().reshape(-1)
            if len(pl_indices) == 0:
                cluster_info = multilayer_cluster_kernels(means3D, normals, False, nlayer_cluster)
            else:
                pl_indice = pl_indices[0].item()
                non_ls_cluster = multilayer_cluster_kernels(means3D[:pl_indice], normals[:pl_indice], False, nlayer_cluster)
                pl_cluster = multilayer_cluster_kernels(means3D[pl_indice:], normals[pl_indice:], True, nlayer_cluster)
                cluster_info = multilayer_merge_clusters(non_ls_cluster, pl_cluster)
    
    for walk_idx in range(num_walks) if not verbose else tqdm(range(num_walks)):
        start_time = time()
        current_stop_idx_s = __sample_stop_idx_s(walk_idx, num_elements, radiosity, radiosity_square, denom, valid_s, verbose)
        if current_stop_idx_s is None:
            break

        latest_estimate_radiosity = (radiosity / denom) if walk_idx > 0 else emissions

        scaled_latest_estimate_radiosity = latest_estimate_radiosity * norm_factors[:, None, :]

        if not use_cluster:
            next, pdf = _C.next_event_estimator(r.randint(0, max_seed), active_sh_degree, max_sh_degree, use_brdf_importance_sampling, channel_clamp_type == 'luminance', means3D, scales, rots, normals, scaled_latest_estimate_radiosity, brdf_coeffs, is_light_source, current_stop_idx_s, means3D, normals, scaled_latest_estimate_radiosity, is_light_source, form_factor_cache, decay_light_source, inverse_falloff_max, min_decay)

            valid_mask = torch.logical_and(next >= 0, ~torch.isnan(pdf))
            # For invalid ones, we at least should add their emissions
            x = emissions[current_stop_idx_s[~valid_mask]]
            radiosity.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x)
            radiosity_square.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x.square())
            denom.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None], torch.ones_like(current_stop_idx_s[~valid_mask][:, None, None], dtype=denom.dtype))
            valid_s.scatter_(0, current_stop_idx_s.long()[~valid_mask], False)
            
            current_stop_idx_s = current_stop_idx_s[valid_mask]
            next = next[valid_mask]
            pdf = pdf[valid_mask]
        else:
            # Populate `cluster_radiosities`
            for cluster_layer_idx, cluster in enumerate(cluster_info):
                if cluster_layer_idx == 0:
                    cluster["cluster_radiosities"] = sum_in_cluster(cluster, latest_estimate_radiosity)
                else:
                    cluster["cluster_radiosities"] = sum_in_cluster(cluster, cluster_info[cluster_layer_idx - 1]["cluster_radiosities"])
            
            # Sample the top layer
            next, pdf = _C.next_event_estimator(r.randint(0, max_seed), active_sh_degree, max_sh_degree, use_brdf_importance_sampling, channel_clamp_type == 'luminance', means3D, scales, rots, normals, latest_estimate_radiosity, brdf_coeffs, is_light_source, current_stop_idx_s, cluster_info[-1]["cluster_xyzs"], cluster_info[-1]["cluster_normals"], cluster_info[-1]["cluster_radiosities"], cluster_info[-1]["cluster_lss"], torch.Tensor([]).int(), decay_light_source, inverse_falloff_max, min_decay)

            valid_mask = torch.logical_and(next >= 0, ~torch.isnan(pdf))
            x = emissions[current_stop_idx_s[~valid_mask]]
            radiosity.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x)
            radiosity_square.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x.square())
            denom.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None], torch.ones_like(current_stop_idx_s[~valid_mask][:, None, None], dtype=denom.dtype))

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
                
                next, pdf = _C.in_cluster_next_event_estimator(r.randint(0, max_seed), active_sh_degree, max_sh_degree, use_brdf_importance_sampling, channel_clamp_type == 'luminance', means3D, normals, brdf_coeffs, is_light_source, current_stop_idx_s, next, pdf, cluster["cluster_end_offset"], cluster["cat_cluster_idx2idx"], cluster["cat_cluster_idx"], sorted_means3D, sorted_normals, sorted_radiosities, sorted_is_ls, form_factor_cache if layer_idx == 0 else torch.Tensor([]).int(), decay_light_source, inverse_falloff_max, min_decay)

                valid_mask = torch.logical_and(next >= 0, ~torch.isnan(pdf))
                x = emissions[current_stop_idx_s[~valid_mask]]
                radiosity.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x)
                radiosity_square.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x.square())
                denom.scatter_add_(0, current_stop_idx_s.long()[~valid_mask][:, None, None], torch.ones_like(current_stop_idx_s[~valid_mask][:, None, None], dtype=denom.dtype))
                
                if not valid_mask.any().item():
                    continue_flag = True
                    break

                current_stop_idx_s = current_stop_idx_s[valid_mask]
                next = next[valid_mask]
                pdf = pdf[valid_mask]
            
            if continue_flag:
                continue
        
        if verbose:
            tqdm.write(f'Elapsed: Estimate Next Stop. {time() - start_time:<.5f}')

        if len(current_stop_idx_s) <= 0:
            continue
        
        weight = torch.reciprocal(pdf + 1E-5)[:, None, None]
        ff = estimator(
            latest_estimate_radiosity, 
            channel_clamp_type == 'luminance', 
            next, 
            current_stop_idx_s, 
            stop_mask=torch.ones_like(current_stop_idx_s).bool(), 
            light_source_mask=is_light_source, 
            form_factor_cache=form_factor_cache, 
            light_source_decay=decay_light_source, 
            inverse_falloff_max=inverse_falloff_max, 
            min_decay=min_decay
        )

        x = emissions[current_stop_idx_s] + weight * ff
        visible_mask = (ff != 0.).any(dim=-1).any(dim=-1)
        packed_emitter_receiver = merge_packed_emitter_receiver(packed_emitter_receiver, next[visible_mask], current_stop_idx_s[visible_mask])
        
        if verbose:
            tqdm.write(f'Elapsed: Calculate One Bounce. {time() - start_time:<.5f}. Visible Ratio: {visible_mask.sum().item()}/{len(visible_mask)}.')
        
        radiosity.scatter_add_(0, current_stop_idx_s.long()[:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x)
        radiosity_square.scatter_add_(0, current_stop_idx_s.long()[:, None, None].expand(-1, emissions.shape[-2], emissions.shape[-1]), x.square())
        denom.scatter_add_(0, current_stop_idx_s.long()[:, None, None], torch.ones_like(current_stop_idx_s[:, None, None], dtype=denom.dtype))

    avg_radiosity = radiosity / denom
    return avg_radiosity, packed_emitter_receiver