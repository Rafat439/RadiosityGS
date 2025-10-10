import torch
import random as r
import numpy as np
import _one_bounce_estimator as _C
from typing import NamedTuple, Optional, Union, Literal

__all__ = [
    'OneBounceEstimator'
]

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

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

def get_bounding_box(
    means3D: torch.Tensor, 
    scales: torch.Tensor, 
    quats: torch.Tensor, 
    geovalues: torch.Tensor
) -> torch.Tensor:
    effective_range = (build_rotation(quats)[:, :, :2] * (torch.sqrt(2.0 * torch.log(geovalues / 0.5358).clamp_min(0.)) * scales)[:, None, :]).norm(dim=-1)
    aabbs = torch.cat([means3D - effective_range, means3D + effective_range], dim=-1)
    return aabbs

def homogeneous(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

class OneBounceEstimator(object):
    def __init__(self, 
        means3D: torch.Tensor, 
        geovalues: torch.Tensor, 
        scales: torch.Tensor, 
        rots: torch.Tensor, 
        normals: torch.Tensor, 
        norm_factors: torch.Tensor, 
        active_sh_degree: int, 
        max_sh_degree: int, 
        brdf_coeffs: torch.Tensor, 
        max_seed: int = 4294967295
    ) -> None:
        super().__init__()
        self.aabbs = get_bounding_box(means3D, scales, rots, geovalues)
        self.handle = _C.optix_build_bvh(self.aabbs)

        self.max_seed = max_seed
        self.means3D = means3D
        self.geovalues = geovalues
        self.scales = scales
        self.rots = rots
        self.normals = normals
        self.norm_factors = norm_factors
        
        self.active_sh_degree = active_sh_degree
        self.max_sh_degree = max_sh_degree
        self.brdf_coeffs = brdf_coeffs
    
    def __del__(self):
        if self.handle is not None:
            _C.optix_release_bvh(*self.handle)
            self.handle = None
    
    def __call__(self, 
        emissions: torch.Tensor, 
        clamp: bool, 
        start_idx_s: torch.Tensor, 
        stop_idx_s: torch.Tensor, 
        stop_mask: torch.Tensor, 
        light_source_mask: torch.Tensor, 
        form_factor_cache: Optional[torch.Tensor] = None, 
        return_type: Union[Literal['walkwise'], Literal['elementwise']] = 'walkwise', 
        light_source_decay: bool = True, 
        inverse_falloff_max: float = 1., 
        min_decay: float = 1e-4
    ) -> torch.Tensor:
        if form_factor_cache is None:
            form_factor_cache = torch.Tensor([]).int()
        seed = r.randint(0, self.max_seed)
        args = [
            seed, self.active_sh_degree, self.max_sh_degree, clamp, 
            self.means3D, self.geovalues, self.scales, self.rots, self.normals, 
            self.norm_factors, emissions, self.brdf_coeffs, 
            start_idx_s, stop_idx_s, stop_mask, light_source_mask, form_factor_cache, return_type == 'walkwise', light_source_decay, inverse_falloff_max, min_decay
        ]
        # cpu_args = cpu_deep_copy_tuple([self.aabbs] + args)
        # try:
        return _C.optix_estimate_one_bounce(
            self.handle[0], self.handle[1], 
            *args
        )
        # except Exception as e:
        #     print("Save to snapshot.")
        #     torch.save(cpu_args, "debug.pth")
        #     raise e