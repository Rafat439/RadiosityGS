#pragma once

#include <cstdio>
#include <torch/torch.h>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> NextEventEstimatorCUDA(
    const unsigned long long seed, 
    const unsigned int active_sh_degree, 
    const unsigned int max_sh_degree, 
    const bool use_brdf_importance_sampling, 
    const bool clamp, 
    const torch::Tensor& means3D, 
    const torch::Tensor& scales, 
    const torch::Tensor& rots, 
    const torch::Tensor& normals, 
    const torch::Tensor& emissions, 
    const torch::Tensor& brdf_coeffs, 
    const torch::Tensor& is_light_source, 
    const torch::Tensor& stop_idx_s, 
    const torch::Tensor& start_means3D, 
    const torch::Tensor& start_normals, 
    const torch::Tensor& start_emissions, 
    const torch::Tensor& start_is_light_source, 
    const torch::Tensor& form_factor_cache, 
    const bool light_source_decay, 
    const float inverse_falloff_max, 
    const float min_decay
);

std::tuple<torch::Tensor, torch::Tensor> InClusterNextEventEstimatorCUDA(
    const unsigned long long seed, 
    const unsigned int active_sh_degree, 
    const unsigned int max_sh_degree, 
    const bool use_brdf_importance_sampling, 
    const bool clamp, 
    const torch::Tensor& means3D, 
    const torch::Tensor& normals, 
    const torch::Tensor& brdf_coeffs, 
    const torch::Tensor& is_light_source, 
    const torch::Tensor& stop_idx_s, 
    const torch::Tensor& next_cluster_idx_s, 
    const torch::Tensor& pdf_s, 
    const torch::Tensor& cluster_idx2offset, 
    const torch::Tensor& sorted_indices, 
    const torch::Tensor& sorted_cluster_idx, 
    const torch::Tensor& sorted_means3D, 
    const torch::Tensor& sorted_normals, 
    const torch::Tensor& sorted_emissions, 
    const torch::Tensor& sorted_is_light_source, 
    const torch::Tensor& form_factor_cache, 
    const bool light_source_decay, 
    const float inverse_falloff_max, 
    const float min_decay
);