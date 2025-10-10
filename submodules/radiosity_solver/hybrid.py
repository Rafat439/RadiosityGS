import torch
import random as r
from time import time
from tqdm import tqdm
from typing import Optional, Tuple, Union, Literal
from submodules.one_bounce_estimator import OneBounceEstimator
from .progressive_refinement import solve_radiosity_by_progressive_refinement
from .monte_carlo import solve_radiosity_by_monte_carlo
from .util import *

@torch.no_grad()
def solve_radiosity_hybrid(
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
    num_walks: int = 16, 
    use_cluster: bool = False, 
    nlayer_cluster: int = 2, 
    use_brdf_importance_sampling: bool = True, 
    channel_clamp_type: Union[Literal['luminance']] = 'luminance', 
    max_seed: int = 4294967295, 
    verbose: bool = False, 
    estimator: OneBounceEstimator = None, 
    cluster_info: Tuple[torch.Tensor, torch.Tensor] = None, 
    form_factor_cache: Optional[torch.Tensor] = None, 
    packed_emitter_receiver: torch.Tensor = None, 
    decay_light_source: bool = True, 
    inverse_falloff_max: float = 1., 
    min_decay: float = 1e-4
) -> torch.Tensor:
    estimator = OneBounceEstimator(means3D, geovalues, scales, rots, normals, norm_factors, active_sh_degree, max_sh_degree, brdf_coeffs, max_seed) if estimator is None else estimator

    # First-round: Direct Illumination (Progressive Refinement)
    # From Light Sources to All Geometric Elements
    radiosity, delta_radiosity, packed_emitter_receiver = solve_radiosity_by_progressive_refinement(
        means3D=means3D, 
        geovalues=geovalues, 
        scales=scales, 
        rots=rots, 
        normals=normals, 
        norm_factors=norm_factors, 
        emissions=emissions, 
        brdf_coeffs=brdf_coeffs, 
        is_light_source=is_light_source, 
        active_sh_degree=active_sh_degree, 
        max_sh_degree=max_sh_degree, 
        max_seed=max_seed, 
        verbose=verbose, 
        estimator=estimator, 
        channel_clamp_type='luminance', 
        form_factor_cache=form_factor_cache, 
        packed_emitter_receiver=packed_emitter_receiver, 
        decay_light_source=decay_light_source, 
        inverse_falloff_max=inverse_falloff_max, 
        min_decay=min_decay
    )

    # Other-rounds: Global Illumination (Monte-Carlo Solving)
    residual_radiosity, packed_emitter_receiver = solve_radiosity_by_monte_carlo(
        means3D=means3D, 
        geovalues=geovalues, 
        scales=scales, 
        rots=rots, 
        normals=normals, 
        norm_factors=norm_factors, 
        emissions=delta_radiosity, 
        brdf_coeffs=brdf_coeffs, 
        is_light_source=is_light_source, 
        active_sh_degree=active_sh_degree, 
        max_sh_degree=max_sh_degree, 
        use_brdf_importance_sampling=use_brdf_importance_sampling, 
        num_walks=num_walks, 
        use_cluster=use_cluster, 
        nlayer_cluster=nlayer_cluster, 
        cluster_info=cluster_info, 
        verbose=verbose, 
        estimator=estimator, 
        channel_clamp_type='luminance', 
        form_factor_cache=form_factor_cache, 
        packed_emitter_receiver=packed_emitter_receiver, 
        decay_light_source=decay_light_source, 
        inverse_falloff_max=inverse_falloff_max, 
        min_decay=min_decay, 
        max_seed=max_seed
    )

    return radiosity + residual_radiosity - delta_radiosity, packed_emitter_receiver