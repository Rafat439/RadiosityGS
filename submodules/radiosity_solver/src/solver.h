#pragma once

#include <cstdio>
#include <torch/torch.h>
#include <torch/extension.h>

std::tuple<unsigned long long, unsigned long long> OptixBuildBVH(
    const torch::Tensor& aabbs
);

void OptixReleaseBVH(
    unsigned long long handle, 
    unsigned long long pointer
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> FusedBackward(
    const unsigned long long handle, 
    const unsigned long long pointer, 
    const unsigned long long seed, 
    const unsigned long long max_seed, 
    const unsigned int active_sh_degree, 
    const unsigned int max_sh_degree, 
    const torch::Tensor& curr_idx_s, 
    const torch::Tensor& next_idx_s, 
    const torch::Tensor& valid_s, 
    const torch::Tensor& means3D, 
    const torch::Tensor& geovalues, 
    const torch::Tensor& scales, 
    const torch::Tensor& rots, 
    const torch::Tensor& normals, 
    const torch::Tensor& norm_factors, 
    const torch::Tensor& is_light_source, 
    const torch::Tensor& A, 
    const torch::Tensor& B, 
    const bool clamp_A, 
    const bool clamp_B, 
    const bool light_source_decay, 
    const torch::Tensor& form_factor_cache, 
    const float inverse_falloff_max, 
    const float min_decay
);