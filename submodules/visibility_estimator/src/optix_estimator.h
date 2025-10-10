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

torch::Tensor OptixVisibilityEstimatorForwardCUDA(
    unsigned long long handle, unsigned long long pointer, 
    const torch::Tensor& means3D, 
    const torch::Tensor& geovalues, 
    const torch::Tensor& scales, 
    const torch::Tensor& rots, 
    const torch::Tensor& start_coords, 
    const torch::Tensor& stop_coords, 
    const float min_decay
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> OptixVisibilityEstimatorBackwardCUDA(
    unsigned long long handle, unsigned long long pointer, 
    const torch::Tensor& means3D, 
    const torch::Tensor& geovalues, 
    const torch::Tensor& scales, 
    const torch::Tensor& rots, 
    const torch::Tensor& start_coords, 
    const torch::Tensor& stop_coords, 
    const torch::Tensor& out_visibility, 
    const torch::Tensor& gradient, 
    const float min_decay
);

#endif