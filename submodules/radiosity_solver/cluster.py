import torch
import math
from typing import List
from pykeops.torch import LazyTensor

def merge_dict(dict_u, dict_v):
    for name in dict_v:
        assert not name in dict_u
        dict_u[name] = dict_v[name]
    return dict_u

@torch.no_grad()
def multilayer_merge_clusters(cluster_Us, cluster_Vs):
    assert len(cluster_Us) == len(cluster_Vs)
    kernels = []
    for cluster_U, cluster_V in zip(cluster_Us, cluster_Vs):
        kernels.append(merge_clusters(cluster_U, cluster_V))
    return kernels

@torch.no_grad()
def multilayer_cluster_kernels(xyzs: torch.Tensor, normals: torch.Tensor, is_ls: bool, layers: int, **kwargs):
    N = len(xyzs)
    K = math.ceil(N ** (1 / layers))
    kernels = []
    for layer in range(layers - 1):
        k = K ** (layers - layer - 1)
        if layer == 0:
            kernels.append(cluster_kernels(xyzs, normals, k, **kwargs))
            kernels[-1]["cluster_lss"] = torch.ones_like(kernels[-1]["cluster_xyzs"][:, 0]).bool().fill_(is_ls)
        else:
            kernels.append(cluster_kernels(kernels[-1]["cluster_xyzs"], kernels[-1]["cluster_normals"], k, **kwargs))
            kernels[-1]["cluster_lss"] = torch.ones_like(kernels[-1]["cluster_xyzs"][:, 0]).bool().fill_(is_ls)
    return kernels

@torch.no_grad()
def cluster_kernels(xyzs: torch.Tensor, normals: torch.Tensor, K: int, Niter: int = 10):
    K = min(K, len(xyzs))
    if K == len(xyzs):
        return dummy_cluster_kernels(xyzs, normals)
    xyzs = xyzs.contiguous()
    normals = normals.contiguous()
    N = len(xyzs)
    device = xyzs.device
    
    indices = torch.multinomial(torch.ones(N, device=device) / N, K)
    # indices = torch.arange(K, dtype=torch.long, device=device)
    c_xyzs = xyzs[indices, :].clone()
    c_normals = normals[indices, :].clone()
    
    i_xyzs = LazyTensor(xyzs.view(N, 1, -1))
    j_c_xyzs = LazyTensor(c_xyzs.view(1, K, -1))

    i_normals = LazyTensor(normals.view(N, 1, -1))
    j_c_normals = LazyTensor(c_normals.view(1, K, -1))
    
    for _ in range(Niter):
        D_ij = ((i_xyzs - j_c_xyzs) ** 2).sum(-1)
        S_ij = (i_normals | j_c_normals).clamp(0., 1.)
        cl = (D_ij / (S_ij + 1E-8)).argmin(dim=1).long().view(-1)
        
        c_xyzs.zero_()
        c_xyzs.scatter_add_(0, cl[:, None].repeat(1, 3), xyzs)
        c_xyzs /= torch.bincount(cl, minlength=K).float().view(K, 1)

        c_normals.zero_()
        c_normals.scatter_add_(0, cl[:, None].repeat(1, 3), normals)
        c_normals[:] = torch.nn.functional.normalize(c_normals, dim=-1, p=2)
    
    cl = cl.int()
    cluster_end_offset = torch.cumsum(torch.bincount(cl, minlength=K).int(), dim=-1).int()
    _sorted_cl = torch.sort(cl)
    cat_cluster_idx2idx = _sorted_cl.indices.int()
    cat_cluster_idx = _sorted_cl.values

    return {
        "idx2cluster_idx": cl.contiguous(), 
        "cat_cluster_idx2idx": cat_cluster_idx2idx.contiguous(), 
        "cat_cluster_idx": cat_cluster_idx.contiguous(), 
        "cluster_end_offset": cluster_end_offset.contiguous(), 
        "cluster_xyzs": c_xyzs.contiguous(), 
        "cluster_normals": c_normals.contiguous()
    }

@torch.no_grad()
def dummy_cluster_kernels(xyzs: torch.Tensor, normals: torch.Tensor):
    cl = torch.arange(len(xyzs), device=xyzs.device, dtype=torch.int)
    cat_cluster_idx2idx = cl
    cat_cluster_idx = cl

    return {
        "idx2cluster_idx": cl, 
        "cat_cluster_idx2idx": cat_cluster_idx2idx, 
        "cat_cluster_idx": cat_cluster_idx, 
        "cluster_end_offset": cl + 1, 
        "cluster_xyzs": xyzs, 
        "cluster_normals": normals
    }

@torch.no_grad()
def merge_clusters(cluster_U, cluster_V):
    num_element_U = len(cluster_U["idx2cluster_idx"])
    num_cluster_U = len(cluster_U["cluster_end_offset"])
    return merge_dict({
        "idx2cluster_idx": torch.cat((cluster_U["idx2cluster_idx"], cluster_V["idx2cluster_idx"] + num_cluster_U)), 
        "cat_cluster_idx2idx": torch.cat((cluster_U["cat_cluster_idx2idx"], cluster_V["cat_cluster_idx2idx"] + num_element_U)), 
        "cat_cluster_idx": torch.cat((cluster_U["cat_cluster_idx"], cluster_V["cat_cluster_idx"] + num_cluster_U)), 
        "cluster_end_offset": torch.cat((cluster_U["cluster_end_offset"], cluster_V["cluster_end_offset"] + num_element_U)), 
        "cluster_xyzs": torch.cat((cluster_U["cluster_xyzs"], cluster_V["cluster_xyzs"])), 
        "cluster_normals": torch.cat((cluster_U["cluster_normals"], cluster_V["cluster_normals"]))
    }, ({ "cluster_lss": torch.cat((cluster_U["cluster_lss"], cluster_V["cluster_lss"])) } if "cluster_lss" in cluster_U else {} ))

@torch.no_grad()
def average_in_cluster(cluster, attributes):
    idx2cluster_idx = cluster["idx2cluster_idx"]
    K = len(cluster["cluster_end_offset"])
    shape = attributes.shape[1:]
    attributes = attributes.reshape(len(attributes), -1)
    avg_attribute = torch.zeros((K, attributes.shape[-1]), device=attributes.device, dtype=attributes.dtype)
    num_elements_in_cluster = torch.bincount(idx2cluster_idx, minlength=K).float()

    avg_attribute.scatter_add_(0, idx2cluster_idx.long()[:, None].repeat(1, attributes.shape[-1]), attributes)
    avg_attribute /= num_elements_in_cluster[:, None].clamp_min_(1)

    return avg_attribute.reshape(-1, *shape)

@torch.no_grad()
def sum_in_cluster(cluster, attributes):
    idx2cluster_idx = cluster["idx2cluster_idx"]
    K = len(cluster["cluster_end_offset"])
    shape = attributes.shape[1:]
    attributes = attributes.reshape(len(attributes), -1)
    sum_attribute = torch.zeros((K, attributes.shape[-1]), device=attributes.device, dtype=attributes.dtype)
    sum_attribute.scatter_add_(0, idx2cluster_idx.long()[:, None].repeat(1, attributes.shape[-1]), attributes)

    return sum_attribute.reshape(-1, *shape)