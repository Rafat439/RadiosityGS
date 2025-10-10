import torch
from time import time
from tqdm import tqdm
from typing import Optional, Union, Literal
from submodules.one_bounce_estimator import OneBounceEstimator
from .util import *

@torch.no_grad()
def solve_radiosity_by_progressive_refinement(
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
    channel_clamp_type: Union[Literal['luminance'], Literal['norm']] = 'luminance', 
    max_seed: int = 4294967295, 
    verbose: bool = False, 
    estimator: OneBounceEstimator = None, 
    form_factor_cache: torch.Tensor = None, 
    packed_emitter_receiver: torch.Tensor = None, 
    decay_light_source: bool = True, 
    inverse_falloff_max: float = 1., 
    min_decay: float = 1e-4
) -> torch.Tensor:
    estimator = OneBounceEstimator(means3D, geovalues, scales, rots, normals, norm_factors, active_sh_degree, max_sh_degree, brdf_coeffs, max_seed) if estimator is None else estimator
    form_factor_cache = form_factor_cache if form_factor_cache is not None else None
    num_elements = means3D.size(0)
    
    radiosity = emissions.clone()
    delta_radiosity = emissions.clone()
    
    assert not torch.isnan(emissions).any() and not torch.isinf(emissions).any(), f'{emissions}'

    def batch_estimate_form_factors(_emissions: torch.Tensor, start_idx_s: torch.Tensor):
        num_query = len(start_idx_s)
        _stop_idx_s = (~is_light_source).reshape(-1).nonzero().int().reshape(-1)
        _start_idx_s = torch.repeat_interleave(start_idx_s, len(_stop_idx_s)).int()
        _stop_idx_s = _stop_idx_s.repeat(num_query)
        _start_idx_s, _stop_idx_s = filter_valid_rays(means3D, normals, geovalues, is_light_source, _start_idx_s, _stop_idx_s, decay_light_source)
        _mask = torch.ones_like(_start_idx_s).bool()
        return estimator(
            _emissions, 
            channel_clamp_type == 'luminance', 
            _start_idx_s, 
            _stop_idx_s, 
            stop_mask=_mask, 
            light_source_mask=is_light_source, 
            form_factor_cache=form_factor_cache, 
            return_type='elementwise', 
            light_source_decay=decay_light_source, 
            inverse_falloff_max=inverse_falloff_max, 
            min_decay=min_decay
        ), _start_idx_s, _stop_idx_s
    
    start_time = time()
    chosen_emit_indices = is_light_source.reshape(-1).nonzero().reshape(-1).int()
    time_chosen = time()
    delta, emit_indices, recv_indices = batch_estimate_form_factors(delta_radiosity, chosen_emit_indices) # (K, N)
    packed_emitter_receiver = merge_packed_emitter_receiver(packed_emitter_receiver, emit_indices, recv_indices)
    time_bounce = time()

    delta_radiosity[chosen_emit_indices] = 0
    radiosity += delta
    delta_radiosity += delta

    time_prop = time()
    if verbose:
        print(f"Direct Illumination: Total time ({time_prop - start_time:<.5f}), Select time ({time_chosen - start_time:<.5f}), Bounce time ({time_bounce - time_chosen:<.5f}), Propagate time ({time_prop - time_bounce:<.5f})")

    return radiosity, delta_radiosity, packed_emitter_receiver