import torch
import _visibility_estimator as _C

__all__ = [
    'estimate_visibility', 'vis_build_bvh', 'vis_release_bvh'
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

def vis_build_bvh(means3D, scales, rots, geovalues):
    aabbs = get_bounding_box(means3D, scales, rots, geovalues)
    return _C.optix_build_bvh(aabbs)

def vis_release_bvh(handle):
    _C.optix_release_bvh(*handle)

def estimate_visibility(handle, means3D: torch.Tensor, geovalues: torch.Tensor, scales: torch.Tensor, rots: torch.Tensor, start_coords_s: torch.Tensor, stop_coords_s: torch.Tensor, min_decay: float):
    return _VisibilityEstimator.apply(
        handle, means3D, geovalues, scales, rots, start_coords_s, stop_coords_s, min_decay
    )

class _VisibilityEstimator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        handle, 
        means3D: torch.Tensor, 
        geovalues: torch.Tensor, 
        scales: torch.Tensor, 
        rots: torch.Tensor, 
        start_coords_s: torch.Tensor, 
        stop_coords_s: torch.Tensor, 
        min_decay: float = 1e-4
    ):
        ctx.handle = handle
        visibility = _C.optix_estimate_visibility_forward(
            ctx.handle[0], ctx.handle[1], 
            means3D, geovalues, scales, rots, 
            start_coords_s, stop_coords_s, min_decay
        )
        visibility[visibility <= min_decay] = 0.0
        ctx.min_decay = min_decay
        ctx.save_for_backward(means3D, geovalues, scales, rots, start_coords_s, stop_coords_s, visibility)
        return visibility
    
    @staticmethod
    def backward(
        ctx, 
        grad_visibility: torch.Tensor
    ):
        means3D, geovalues, scales, rots, start_coords_s, stop_coords_s, visibility = ctx.saved_tensors

        dL_dmeans3D, dL_dgeovalues, dL_dscales, dL_drots, dL_dstart_coords, dL_dstop_coords = _C.optix_estimate_visibility_backward(
            ctx.handle[0], ctx.handle[1], 
            means3D, geovalues, scales, rots, 
            start_coords_s, stop_coords_s, visibility, grad_visibility, ctx.min_decay
        )

        ctx.handle = None

        return (
            None, dL_dmeans3D, dL_dgeovalues, dL_dscales, dL_drots, dL_dstart_coords, dL_dstop_coords
        )