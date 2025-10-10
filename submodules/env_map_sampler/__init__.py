import torch
import numpy as np
from torch.nn import functional as F
from typing import Union, Literal

__all__ = [
    "sample_env_map"
]

def estimateQuaternion(u: torch.Tensor, v: torch.Tensor):
    xyz = torch.cross(u, v, dim=-1)
    w = 1 + (u * v).sum(dim=-1, keepdim=True)
    rot = torch.cat((w, xyz), dim=-1)
    return torch.nn.functional.normalize(rot)

def sample_env_map(
    env_map: torch.Tensor, 
    radius: float = 1000, 
    num_samples: int = 16, 
    opaque_geovalue: float = 6.0
):
    assert len(env_map.shape) == 3
    assert env_map.shape[0] == 1 or env_map.shape[0] == 3
    
    _, H, W = env_map.shape
    i = torch.linspace(0, H - 1, num_samples, device=env_map.device)
    j = torch.linspace(0, W - 1, 2 * num_samples, device=env_map.device)
    jj, ii = torch.meshgrid(j, i, indexing='xy')

    u = jj / W
    v = ii / H
    
    theta = u * 2 * torch.pi
    phi = v * torch.pi

    means3D = torch.stack([
        radius * torch.sin(phi) * torch.cos(theta), 
        radius * torch.cos(phi), 
        radius * torch.sin(phi) * torch.sin(theta)
    ], dim=-1) # (2 * N, N, 3)

    scales = torch.stack([
        torch.ones_like(theta) * radius / (1.414 * num_samples), 
        torch.ones_like(theta) * radius / (1.414 * num_samples)
    ], dim=-1) # (2 * N, N, 2)
    geovalues = torch.ones_like(theta)[..., None] * opaque_geovalue # (2 * N, N, 1)
    normals = -torch.nn.functional.normalize(means3D, p=2, dim=-1)
    from_normals = torch.zeros_like(normals)
    from_normals[:, -1] = 1
    rots = estimateQuaternion(from_normals, normals)

    grid_x = u * 2 - 1
    grid_y = v * 2 - 1
    coords = torch.stack([grid_x, grid_y], dim=-1)
    # print(coords.shape)
    env_map = torch.nn.functional.interpolate(env_map[None], coords.shape[:2], mode='bilinear', align_corners=False, antialias=True)[0]
    emissions = torch.nn.functional.grid_sample(
        env_map[None], coords[None], mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0) # (2 * N, N, -1)

    # azimuth, polar = torch.meshgrid(
    #     torch.linspace(0, 2 * torch.pi, 2 * num_samples, device=env_map.device), 
    #     torch.linspace(0, torch.pi, num_samples, device=env_map.device), 
    #     indexing="xy"
    # )
    
    # theta = azimuth
    # v = polar / torch.pi
    # phi = torch.arccos(1 - 2 * v)
    # means3D = torch.stack([
    #     radius * torch.sin(phi) * torch.cos(torch.pi - theta), 
    #     radius * torch.cos(phi), 
    #     radius * torch.sin(phi) * torch.sin(torch.pi - theta)
    # ], dim=-1) # (2 * N, N, 3)
    # scales = torch.stack([
    #     torch.ones_like(theta) * radius / (1.414 * num_samples), 
    #     torch.ones_like(theta) * radius / (1.414 * num_samples)
    # ], dim=-1) # (2 * N, N, 2)
    # geovalues = torch.ones_like(theta)[..., None] * opaque_geovalue # (2 * N, N, 1)
    # normals = -torch.nn.functional.normalize(means3D, p=2, dim=-1)
    # from_normals = torch.zeros_like(normals)
    # from_normals[:, -1] = 1
    # rots = estimateQuaternion(from_normals, normals)
    
    # u = (theta / (2 * torch.pi)) * 2 - 1
    # v = (1 - (phi / torch.pi)) * 2 - 1
    # coords = torch.stack((u, v), dim=-1)
    # assert 2 * coords.shape[0] == coords.shape[1], f'{coords.shape}'
    # # print(env_map.shape, coords.shape)
    # env_map = torch.nn.functional.interpolate(env_map[None], coords.shape[:2], mode='bilinear', align_corners=False, antialias=True)[0]
    
    # emissions = torch.nn.functional.grid_sample(
    #     env_map[None], coords[None], mode='bilinear', align_corners=False
    # ).squeeze(0).permute(1, 2, 0) # (2 * N, N, -1)
    return {
        "means3D": means3D.reshape(-1, 3), 
        "scales": scales.reshape(-1, 2), 
        "geovalues": geovalues.reshape(-1, 1), 
        "rots": rots.reshape(-1, 4), 
        "emissions": emissions.reshape(-1, env_map.size(0))
    }