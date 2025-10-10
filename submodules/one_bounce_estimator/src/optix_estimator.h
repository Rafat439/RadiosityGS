#ifndef CUDA_OPTIX_ESTIMATOR_H_INCLUDED
#define CUDA_OPTIX_ESTIMATOR_H_INCLUDED

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

torch::Tensor OptixOneBounceEstimatorCUDA(
    unsigned long long handle, unsigned long long pointer, 
    unsigned long long seed, 
    unsigned int active_sh_degree, 
    unsigned int max_sh_degree, 
    bool clamp, 
    const torch::Tensor& means3D, 
    const torch::Tensor& geovalues, 
    const torch::Tensor& scales, 
    const torch::Tensor& rots, 
    const torch::Tensor& normals, 
    const torch::Tensor& norm_factors, 
    const torch::Tensor& emissions, 
    const torch::Tensor& brdf_coeffs, 
    const torch::Tensor& start_idx_s, 
    const torch::Tensor& stop_idx_s, 
    const torch::Tensor& stop_mask, 
    const torch::Tensor& light_source_mask, 
    const torch::Tensor& form_factor_cache, 
    const bool return_type, 
    const bool light_source_decay, 
    const float inverse_falloff_max, 
    const float min_decay
);

#endif