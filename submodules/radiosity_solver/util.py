import torch
import numpy as np

def filter_valid_rays(
    means3D, normals, geovalues, is_pls, start_idx_s, stop_idx_s, decay_light_source=True
) -> torch.Tensor:
    # return start_idx_s, stop_idx_s
    center_ray_dir = torch.nn.functional.normalize(means3D[stop_idx_s] - means3D[start_idx_s])
    center_start_angle_factor = torch.where(is_pls.squeeze()[start_idx_s], 1.0, (normals[start_idx_s] * center_ray_dir).sum(dim=-1))
    center_stop_angle_factor = torch.where(is_pls.squeeze()[stop_idx_s], 1.0, -(normals[stop_idx_s] * center_ray_dir).sum(dim=-1))
    center_angle_factor = center_start_angle_factor * center_stop_angle_factor
    if decay_light_source:
        center_angle_factor = center_angle_factor * (1.0 / (torch.square(torch.linalg.vector_norm(means3D[stop_idx_s] - means3D[start_idx_s], ord=2, dim=-1)) + 1E-8))

    mask = torch.logical_and(torch.logical_and(center_start_angle_factor > 0, center_stop_angle_factor > 0), center_angle_factor > 1E-4)
    mask &= torch.logical_and(geovalues.squeeze()[start_idx_s] > 0.5358, geovalues.squeeze()[stop_idx_s] > 0.5358)
    return start_idx_s[mask].contiguous(), stop_idx_s[mask].contiguous()

def pack_emitter_receiver(
    emit_indices: torch.Tensor, recv_indices: torch.Tensor
) -> torch.Tensor:
    return emit_indices.long().reshape(-1) << 32 | recv_indices.long().reshape(-1)

def unpack_emitter_receiver(
    packed_indices: torch.Tensor
) -> torch.Tensor:
    return packed_indices >> 32, packed_indices & 0xFFFFFFFF

def merge_packed_emitter_receiver(
    packed_indices: torch.Tensor, emit_indices: torch.Tensor, recv_indices: torch.Tensor
) -> torch.Tensor:
    if packed_indices is None:
        return pack_emitter_receiver(emit_indices, recv_indices)
    else:
        new_packed_indices = pack_emitter_receiver(emit_indices, recv_indices)
        return torch.unique(torch.cat((packed_indices, new_packed_indices)))

class LambdaActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, geovalues: torch.Tensor, scales: torch.Tensor):
        g = 0.03279 * torch.pow(geovalues.clamp_min(0.), 3.4)
        # From https://en.wikipedia.org/wiki/Exponential_integral
        y = 2 * np.pi / 3.4 * (np.euler_gamma + torch.log(g) + torch.pow(torch.pow(torch.log((1 + g) * (0.56146 / g + 0.65)), -7.7) + torch.pow(g, 4) * torch.exp(7.7 * g) * torch.pow(2 + g, 3.7), -0.13))
        ctx.save_for_backward(geovalues, scales, g, y)
        return y * torch.prod(scales, dim=-1, keepdim=True)
    
    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        geovalues, scales, g, y = ctx.saved_tensors
        grad_scales = grad * y * torch.flip(scales, dims=(-1,))
        grad_geovalues = grad * torch.prod(scales, dim=-1, keepdim=True) * 2 * np.pi / 3.4 * ((1 - torch.exp(-g)) / (g + 1E-8)) * 0.03279 * 3.4 * torch.pow(geovalues.clamp_min(0.), 2.4)
        return grad_geovalues, grad_scales

def non_ls_lambda_activation(geovalues: torch.Tensor, scales: torch.Tensor):
    return LambdaActivation.apply(geovalues, scales)

def lambda_activation(geovalues: torch.Tensor, scales: torch.Tensor, is_ls: torch.Tensor):
    g = 0.03279 * torch.pow(geovalues.clamp_min(0.), 3.4)
    return torch.where(is_ls, torch.ones_like(geovalues), 2 * np.pi / 3.4 * (np.euler_gamma + torch.log(g) + torch.pow(torch.pow(torch.log((1 + g) * (0.56146 / g + 0.65)), -7.7) + torch.pow(g, 4) * torch.exp(7.7 * g) * torch.pow(2 + g, 3.7), -0.13)) * torch.prod(scales, dim=-1, keepdim=True))