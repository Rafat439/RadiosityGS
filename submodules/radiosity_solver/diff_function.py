import torch
from typing import Tuple, Union, Literal, Optional
from .hybrid import *
from .monte_carlo import *
from .progressive_refinement import *

import warnings
from time import time
from .cluster import *
from dataclasses import dataclass
import _radiosity_solver_aux as _C # type: ignore

@dataclass
class RadiosityPropagationSettings:
    solver_type                 : Union[Literal['PR'], Literal['MC'], Literal['hybrid']]
    debug                       : bool                                                    = False

    '''Common Properties'''
    max_seed                    : int                                                     = 4294967295
    active_sh_degree            : int                                                     = 0
    max_sh_degree               : int                                                     = 9
    directional_light_source    : bool                                                    = False
    inverse_falloff_max         : float                                                   = 1.
    min_decay                   : float                                                   = 1e-4

    '''Monte-Carlo Properties'''
    num_walks                   : int                                                     = 128
    use_cluster                 : bool                                                    = True
    nlayer_cluster              : int                                                     = 2

    '''Gradient Properties'''
    gradient_num_walks          : int                                                     = 128

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

def apply_solver(
    means3D             : torch.Tensor, 
    geovalues           : torch.Tensor, 
    scales              : torch.Tensor, 
    rots                : torch.Tensor, 
    normals             : torch.Tensor, 
    norm_factors        : torch.Tensor, 
    emissions           : torch.Tensor, 
    brdf_coeffs         : torch.Tensor, 
    is_light_source     : torch.Tensor, 
    settings            : RadiosityPropagationSettings, 
    estimator           : OneBounceEstimator, 
    channel_clamp_type  : Union[Literal['luminance'], Literal['norm']], 
    use_brdf_is         : bool, 
    form_factor_cache   : Optional[torch.Tensor] = None, 
    solver_type_override: Optional[Union[Literal['PR'], Literal['MC'], Literal['hybrid']]] = None, 
    num_walks_override  : Optional[int] = None, 
    cluster_info        : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
    verbose             : bool = False, 
    **kwargs
) -> torch.Tensor:
    common_args = { 'means3D': means3D, 'geovalues': geovalues, 'scales': scales, 'rots': rots, 'normals': normals, 'emissions': emissions, 'brdf_coeffs': brdf_coeffs, 'is_light_source': is_light_source, 'norm_factors': norm_factors, 'channel_clamp_type': channel_clamp_type, 'active_sh_degree': settings.active_sh_degree, 'max_sh_degree': settings.max_sh_degree, 'max_seed': settings.max_seed, 'decay_light_source': not settings.directional_light_source, 'inverse_falloff_max': settings.inverse_falloff_max, 'min_decay': settings.min_decay }
    solver_type = settings.solver_type if solver_type_override is None else solver_type_override
    num_walks = settings.num_walks if num_walks_override is None else num_walks_override

    if solver_type == 'PR':
        if verbose:
            print("Apply Progressive Refinement Solver ...")
        radiosity, _, packed_emitter_receiver = solve_radiosity_by_progressive_refinement(
            verbose=verbose, 
            estimator=estimator, 
            form_factor_cache=form_factor_cache, 
            **common_args, **kwargs
        )
    elif solver_type == 'MC':
        if verbose:
            print("Apply Native Monte-Carlo Solver ...")
        radiosity, packed_emitter_receiver = solve_radiosity_by_monte_carlo(
            num_walks=num_walks, 
            use_brdf_importance_sampling=use_brdf_is, 
            verbose=verbose, 
            estimator=estimator, 
            form_factor_cache=form_factor_cache, 
            use_cluster=settings.use_cluster, 
            cluster_info=cluster_info, 
            nlayer_cluster=settings.nlayer_cluster, 
            **common_args, **kwargs
        )
    elif solver_type == 'hybrid':
        if verbose:
            print("Apply Hybrid Monte-Carlo Solver ...")
        radiosity, packed_emitter_receiver = solve_radiosity_hybrid(
            num_walks=num_walks, 
            verbose=verbose, 
            estimator=estimator, 
            use_brdf_importance_sampling=use_brdf_is, 
            form_factor_cache=form_factor_cache, 
            use_cluster=settings.use_cluster, 
            cluster_info=cluster_info, 
            nlayer_cluster=settings.nlayer_cluster, 
            **common_args, **kwargs
        )
    
    return radiosity, packed_emitter_receiver

class _PropagateRadiosity(torch.autograd.Function):
    def forward(
        ctx, 
        means3D             : torch.Tensor, 
        geovalues           : torch.Tensor, 
        scales              : torch.Tensor, 
        rots                : torch.Tensor, 
        normals             : torch.Tensor, 
        norm_factors        : torch.Tensor, 
        emissions           : torch.Tensor, 
        brdf_coeffs         : torch.Tensor, 
        is_light_source     : torch.Tensor, 
        settings            : RadiosityPropagationSettings, 
        form_factor_cache   : Optional[torch.Tensor]
    ):  
        start = time()
        cluster_info = None
        non_ls_cluster = None
        if settings.use_cluster:
            # Assume `is_light_source` is sorted.
            pl_indices = is_light_source.nonzero().reshape(-1)
            if len(pl_indices) == 0:
                non_ls_cluster = cluster_info = multilayer_cluster_kernels(means3D, normals, False, settings.nlayer_cluster)
            else:
                pl_indice = pl_indices[0].item()
                non_ls_cluster = multilayer_cluster_kernels(means3D[:pl_indice], normals[:pl_indice], False, settings.nlayer_cluster)
                pl_cluster = multilayer_cluster_kernels(means3D[pl_indice:], normals[pl_indice:], True, settings.nlayer_cluster)
                cluster_info = multilayer_merge_clusters(non_ls_cluster, pl_cluster)

        estimator = OneBounceEstimator(means3D, geovalues, scales, rots, normals, norm_factors, settings.active_sh_degree, settings.max_sh_degree, brdf_coeffs, settings.max_seed)
        
        if settings.debug:
            print(f'BVH Building Time Elapsed: {time() - start:<.4f}')
        
        direct_light_only = settings.solver_type == 'PR'
        if settings.debug:
            print(f'Invoke direct-light only optimization? {direct_light_only}')
        
        radiosity, packed_emitter_receiver = apply_solver(
            means3D=means3D, geovalues=geovalues, scales=scales, rots=rots, normals=normals, emissions=emissions, norm_factors=norm_factors, brdf_coeffs=brdf_coeffs, is_light_source=is_light_source, 
            settings=settings, estimator=estimator, channel_clamp_type='luminance', use_brdf_is=False, verbose=settings.debug, form_factor_cache=form_factor_cache, cluster_info=cluster_info
        )
        
        if settings.debug:
            print(f'Forward Radiosity Time Elapsed: {time() - start:<.4f}')
        
        ctx.settings = settings
        ctx.estimator = estimator
        ctx.non_ls_cluster = non_ls_cluster
        ctx.cluster_info = cluster_info
        ctx.direct_light_only = direct_light_only
        ctx.save_for_backward(means3D, geovalues, scales, rots, normals, norm_factors, emissions, brdf_coeffs, radiosity, is_light_source, form_factor_cache, packed_emitter_receiver)
        
        return radiosity
    
    def backward(ctx, grad_radiosity: torch.Tensor):
        settings = ctx.settings
        means3D, geovalues, scales, rots, normals, norm_factors, emissions, brdf_coeffs, radiosity, is_light_source, form_factor_cache, packed_emitter_receiver = ctx.saved_tensors
        
        estimator = ctx.estimator
        cluster_info = ctx.cluster_info
        non_ls_cluster = ctx.non_ls_cluster
        
        direct_light_only = ctx.direct_light_only
        if settings.debug:
            print(f"Direct-only Optimization? {direct_light_only}")
        chosen_emit_indices = None
        
        brdf_coeff_mul_grad_radiosity = grad_radiosity * brdf_coeffs
        
        start = time()
        
        if direct_light_only:
            # Direct Illumination
            num_elements = len(means3D)
            chosen_emit_indices = is_light_source.reshape(-1).nonzero().int().reshape(-1)
            _start_idx_s = (~is_light_source).reshape(-1).nonzero().int().reshape(-1)
            _stop_idx_s = torch.repeat_interleave(chosen_emit_indices, len(_start_idx_s)).int()
            _start_idx_s = _start_idx_s.repeat(len(chosen_emit_indices))
            _start_idx_s, _stop_idx_s = filter_valid_rays(means3D, normals, geovalues, is_light_source, _start_idx_s, _stop_idx_s, not settings.directional_light_source)
            _mask = torch.ones_like(_start_idx_s).bool()
            A = brdf_coeff_mul_grad_radiosity.clone()
            if len(_start_idx_s) > 0 and len(_stop_idx_s) > 0:
                assert brdf_coeff_mul_grad_radiosity.shape == emissions.shape
                A[chosen_emit_indices] += estimator(
                    brdf_coeff_mul_grad_radiosity, 
                    False, 
                    _start_idx_s, 
                    _stop_idx_s, 
                    _mask, 
                    light_source_mask=is_light_source, 
                    form_factor_cache=form_factor_cache, 
                    return_type='elementwise', 
                    light_source_decay=not settings.directional_light_source, 
                    inverse_falloff_max=settings.inverse_falloff_max, 
                    min_decay=settings.min_decay
                )[chosen_emit_indices]
        else:
            # Total Illumination
            non_light_source = ~is_light_source.squeeze()
            _A = apply_solver(
                means3D=means3D[non_light_source], geovalues=geovalues[non_light_source], scales=scales[non_light_source], rots=rots[non_light_source], normals=normals[non_light_source], emissions=brdf_coeff_mul_grad_radiosity[non_light_source], brdf_coeffs=brdf_coeffs[non_light_source], norm_factors=norm_factors[non_light_source], is_light_source=is_light_source[non_light_source], 
                settings=settings, estimator=None, channel_clamp_type='norm', use_brdf_is=False, solver_type_override='MC', num_walks_override=settings.gradient_num_walks, verbose=settings.debug, form_factor_cache=form_factor_cache, cluster_info=None
            )[0]
            A = torch.zeros_like(brdf_coeff_mul_grad_radiosity)
            A[non_light_source] = _A

            chosen_emit_indices = is_light_source.reshape(-1).nonzero().reshape(-1).int()
            _start_idx_s = ((~is_light_source).reshape(-1).nonzero().reshape(-1).int()).repeat(len(chosen_emit_indices))
            _stop_idx_s = torch.repeat_interleave(chosen_emit_indices, (~is_light_source).sum().item()).int()
            _mask = torch.ones_like(_start_idx_s).bool()
            A[chosen_emit_indices] += estimator(
                A, 
                False, 
                _start_idx_s, 
                _stop_idx_s, 
                _mask, 
                light_source_mask=is_light_source, 
                form_factor_cache=form_factor_cache, 
                return_type='elementwise', 
                light_source_decay=not settings.directional_light_source, 
                inverse_falloff_max=settings.inverse_falloff_max, 
                min_decay=settings.min_decay
            )[chosen_emit_indices]

        grad_emissions = (A / brdf_coeffs)
        grad_emissions = torch.nan_to_num(grad_emissions)
        if settings.debug:
            print(f'dL / demission, A Elapsed: {time() - start:<.4f}')
        
        grad_brdf_coefficients = (radiosity - emissions) / brdf_coeffs * grad_emissions
        grad_brdf_coefficients = torch.nan_to_num(grad_brdf_coefficients)
        if settings.debug:
            print(f'dL / dbrdf Elapsed: {time() - start:<.4f}')
        
        # The calc. of geometry gradients need to enumerate all visible pairs.
        # However, as in the training where we gradually increase the number of walks, 
        # not all visible pairs are recorded at early stages.
        # We test different strategies, such as using complicated MC process to 
        # randomly select pairs to estimate the summation but find that simply reusing 
        # the visible pairs produce similar results and it is much simplier and faster.
        se_emitter, se_receiver = unpack_emitter_receiver(torch.sort(packed_emitter_receiver).values)
        valid_mask = ~is_light_source[se_receiver].squeeze()
        se_emitter = se_emitter[valid_mask]
        se_receiver = se_receiver[valid_mask]
        
        dL_dmeans3D, dL_dgeovalues, dL_dscales, dL_drots, dL_dnormals, dL_dnorm_factors = _C.fused_backward(
            estimator.handle[0], estimator.handle[1], 
            r.randint(0, estimator.max_seed), estimator.max_seed, 
            settings.active_sh_degree, settings.max_sh_degree, 
            se_receiver.int()[None], se_emitter.int()[None], torch.ones((1, len(se_emitter)), dtype=torch.bool, device="cuda"), 
            means3D, geovalues, scales, rots, normals, norm_factors, is_light_source, 
            A, radiosity, 
            False, True, not settings.directional_light_source, form_factor_cache, settings.inverse_falloff_max, settings.min_decay
        )
        
        if settings.debug:
            print(f'Backward Elapsed: {time() - start:<.4f}')
        
            torch.save([
                means3D, geovalues, scales, rots, normals, brdf_coeffs, 
                grad_radiosity, A, 
                dL_dmeans3D, 
                dL_dgeovalues, 
                dL_dscales, 
                dL_drots, 
                dL_dnormals, 
                dL_dnorm_factors, 
                grad_emissions, 
                grad_brdf_coefficients
            ], 'grad_snapshot.pth')
        
        return (
            dL_dmeans3D, 
            dL_dgeovalues, 
            dL_dscales, 
            dL_drots, 
            dL_dnormals, 
            dL_dnorm_factors, 
            grad_emissions, 
            grad_brdf_coefficients, 
            None, 
            None, 
            None
        )
    
from copy import deepcopy

class RadiosityPropagater(torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = deepcopy(settings)
    
    def forward(self, 
        means3D             : torch.Tensor, 
        geovalues           : torch.Tensor, 
        scales              : torch.Tensor, 
        rots                : torch.Tensor, 
        norm_factors        : torch.Tensor, 
        emissions           : torch.Tensor, 
        brdf_coeffs         : torch.Tensor, 
        is_light_source     : torch.Tensor, 
    ):
        _uniq_consec_is_ls = torch.unique_consecutive(is_light_source.reshape(-1))
        assert (len(_uniq_consec_is_ls) == 1) or (len(_uniq_consec_is_ls) == 2 and (is_light_source[0] == False).item()), f"By convention, light sources must be put at the end of the list. {_uniq_consec_is_ls}"
        assert brdf_coeffs.shape == emissions.shape

        normals = build_scaling_rotation(torch.cat([scales, torch.ones_like(scales[:, :1])], dim=-1), rots).permute(0, 2, 1)[:, -1, :]
        
        # Cache the form factor between light sources and other elements.
        # A full cache between any two could cause OOM.
        # - store the number of element at the first place.
        first_pl_indice = is_light_source.reshape(-1).nonzero().reshape(-1)[0]
        n_non_ls = first_pl_indice
        n_ls = len(means3D) - n_non_ls
        form_factor_cache = torch.cat((n_non_ls * torch.ones((1,), dtype=torch.int, device="cuda"), n_ls * torch.ones((1,), dtype=torch.int, device="cuda"), -torch.ones((n_non_ls * n_ls, ), dtype=torch.int, device="cuda")))
        
        return _PropagateRadiosity.apply(
            means3D, 
            geovalues, 
            scales, 
            rots, 
            normals, 
            norm_factors, 
            emissions, 
            brdf_coeffs, 
            is_light_source, 
            self.settings, 
            form_factor_cache
        )