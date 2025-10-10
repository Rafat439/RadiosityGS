#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include "next_event.h"
#include "auxiliary.h"

#define BLOCK_SIZE 256
#define EXPANSION 4

__forceinline__ __device__ int sampleWithWeightedProbabilityBinarySearch(
    const unsigned long long& seed, 
    const unsigned long long& subsequence, 
    const unsigned long long& offset, 
    const float* __restrict__ unnorm_weights, 
    const float sum_unnorm_weights, 
    const int length
) {
    curandState state;
    curand_init(seed, subsequence, offset, &state);
    float random_val = curand_uniform(&state);

    int left = 0, right = length - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;

        float val = unnorm_weights[mid] / sum_unnorm_weights;
        if ( random_val > val ) left = mid + 1;
        else right = mid;
    }

    return left;
}

__global__ void estimateNextEventCUDA(
    const unsigned long long seed, 
    const unsigned int active_sh_degree, 
    const unsigned int max_sh_degree, 
    const bool use_brdf_importance_sampling, 
    const unsigned int clamp, 
    const unsigned int N, const unsigned int M, 
    const bool cluster_sampling, 
    const glm::vec3* __restrict__ means3D, 
    const glm::vec2* __restrict__ scales, 
    const glm::vec4* __restrict__ rots, 
    const glm::vec3* __restrict__ normals, 
    const glm::vec3* __restrict__ emissions, 
    const glm::vec3* __restrict__ brdf_coeffs, 
    const bool* __restrict__ is_light_source, 
    const int* __restrict__ stop_idx_s, 
    const glm::vec3* __restrict__ start_means3D, 
    const glm::vec3* __restrict__ start_normals, 
    const glm::vec3* __restrict__ start_emissions, 
    const bool* __restrict__ start_is_light_source, 
    int* __restrict__ out_idx_s, 
    float* __restrict__ out_pdf_s, 
    int* __restrict__ form_factor_cache, 
    const bool light_source_decay, 
    const float inverse_falloff_max, 
    const float min_decay
) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool done = idx >= N;

    const unsigned int SH_STRIDE = (max_sh_degree + 1) * (max_sh_degree + 1);
    int stop_idx = -1;
    register glm::vec3 stop_mean3D, stop_normal;
    bool is_ls;
    if ( !done ) {
        stop_idx = stop_idx_s[idx];
        stop_mean3D = means3D[stop_idx];
        stop_normal = normals[stop_idx];
        is_ls = is_light_source[stop_idx];
    }

    __shared__ glm::vec3 collected_means3D[EXPANSION * BLOCK_SIZE];
    __shared__ glm::vec3 collected_normals[EXPANSION * BLOCK_SIZE];
    __shared__ float collected_emissions[EXPANSION * BLOCK_SIZE];
    __shared__ bool collected_is_light_sources[EXPANSION * BLOCK_SIZE];

    float unnormalized_probs[EXPANSION * BLOCK_SIZE] = {0};
    float accum_unnormalized_probs[EXPANSION * BLOCK_SIZE] = {0};
    int current_idx = -1;
    float current_unnormalized_prob = 0.0F;
    float current_equivalent_prob = 0.0F;

    const int rounds = (M + (EXPANSION * BLOCK_SIZE) - 1) / (EXPANSION * BLOCK_SIZE);

    #pragma unroll 64
    for ( int r = 0; r < rounds; ++r ) {
        int num_done = __syncthreads_count(done);
        if ( num_done == BLOCK_SIZE )
            break;

        #pragma unroll 4
        for ( int f = 0; f < EXPANSION; ++f ) {
            const int sub_idx = (threadIdx.x * EXPANSION) + f;
            const int fetch_idx = r * (EXPANSION * BLOCK_SIZE) + sub_idx;
            if ( fetch_idx < M ) {
                collected_means3D[sub_idx] = start_means3D[fetch_idx];
                collected_normals[sub_idx] = start_normals[fetch_idx];
                // Only 0-th Degree for Speeding Up
                collected_emissions[sub_idx] = fabs(clampBaseProb(clamp? 0 : 3, computeOutRadianceFromSH(0, (start_emissions + fetch_idx * SH_STRIDE), glm::vec3(0.0F), clamp)));
                collected_is_light_sources[sub_idx] = start_is_light_source[fetch_idx];
            }
        }
        __syncthreads();

        const int size = min(EXPANSION * BLOCK_SIZE, M - r * EXPANSION * BLOCK_SIZE);
        bool any_valid = false;
        float current_sum_unnormalized_prob = 0.0F;

        #pragma unroll 64
        for ( int i = 0; !done && i < size; ++i ) {
            const auto start_idx = i + r * BLOCK_SIZE * EXPANSION;
            if ( !cluster_sampling && stop_idx == start_idx ) {
                unnormalized_probs[i] = 0.0F;
                accum_unnormalized_probs[i] = ((i > 0)? accum_unnormalized_probs[i - 1] : 0.0F);
                continue;
            }
            const glm::vec3 start_mean3D = collected_means3D[i];
            const glm::vec3 start_normal = collected_normals[i];
            const glm::vec3 dir = glm::normalize(stop_mean3D - start_mean3D);
            const auto length = glm::length(stop_mean3D - start_mean3D);
            float fused_prob = 0.0F;
            const auto stop_angle_factor = (is_ls? 1.0F : -glm::dot(stop_normal, dir));
            const auto start_angle_factor = (collected_is_light_sources[i]? 1.0F : glm::dot(start_normal, dir));
            const auto inverse_square_law = (light_source_decay || (!is_ls && !collected_is_light_sources[i]))? min(inverse_falloff_max, 1.0F / (length * length + 1E-8F)) : 1.0F;
            if ( (cluster_sampling && start_angle_factor > 0.0F && stop_angle_factor > 0.0F && inverse_square_law > 0.0F) || (start_angle_factor > min_decay && stop_angle_factor > min_decay && inverse_square_law > min_decay) ) {
                // Rough Form Factor
                fused_prob = inverse_square_law * start_angle_factor * stop_angle_factor * collected_emissions[i];

                if ( fused_prob < min_decay ) fused_prob = 0.0F;
                else any_valid = true;
            }
            unnormalized_probs[i] = fused_prob;
            accum_unnormalized_probs[i] = ((i > 0)? accum_unnormalized_probs[i - 1] : 0.0F) + fused_prob;
            current_sum_unnormalized_prob += fused_prob;
        }

        if ( !done && any_valid ) {
            // Sample from this batch
            auto res = sampleWithWeightedProbabilityBinarySearch(seed, idx, r * 2, accum_unnormalized_probs, current_sum_unnormalized_prob, size);
            const int selected_idx = res + r * BLOCK_SIZE * EXPANSION;
            const float selected_unnorm_prob = unnormalized_probs[res];
            if ( current_idx < 0 ) {
                current_idx = selected_idx;
                current_unnormalized_prob = selected_unnorm_prob;
                current_equivalent_prob = current_sum_unnormalized_prob;
            } else {
                int two_idx[] = {current_idx, selected_idx};
                float two_unnormalized_prob[] = { current_unnormalized_prob, selected_unnorm_prob };
                float two_accum_unnormalized_prob[] = { current_equivalent_prob, current_equivalent_prob + current_sum_unnormalized_prob };
                float two_sum_accum = current_equivalent_prob + current_sum_unnormalized_prob;
                auto res = sampleWithWeightedProbabilityBinarySearch(seed, idx, r * 2 + 1, two_accum_unnormalized_prob, two_sum_accum, 2);

                current_idx = two_idx[res];
                current_unnormalized_prob = two_unnormalized_prob[res];
                current_equivalent_prob = two_sum_accum;
            }
        }
    }

    if ( idx < N ) {
        out_idx_s[idx] = current_idx;
        out_pdf_s[idx] = current_unnormalized_prob / (current_equivalent_prob + 1E-8F);
    }
}

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
) {
    const unsigned int N = stop_idx_s.size(0);
    const unsigned int M = start_means3D.size(0);
    const bool cluster_sampling = start_means3D.size(0) != means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts = means3D.options().dtype(torch::kInt);
    auto out_idx_s = torch::full({N}, -1, int_opts);
    auto out_pdf_s = torch::full({N}, 0.0F, float_opts);
    estimateNextEventCUDA <<< (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (
        seed, active_sh_degree, max_sh_degree, use_brdf_importance_sampling, clamp, N, M, cluster_sampling, 
        (glm::vec3*)means3D.contiguous().data_ptr<float>(), 
        (glm::vec2*)scales.contiguous().data_ptr<float>(), 
        (glm::vec4*)rots.contiguous().data_ptr<float>(), 
        (glm::vec3*)normals.contiguous().data_ptr<float>(), 
        (glm::vec3*)emissions.contiguous().data_ptr<float>(), 
        (glm::vec3*)brdf_coeffs.contiguous().data_ptr<float>(), 
        is_light_source.contiguous().data_ptr<bool>(), 
        stop_idx_s.contiguous().data_ptr<int>(), 
        (glm::vec3*)start_means3D.contiguous().data_ptr<float>(), 
        (glm::vec3*)start_normals.contiguous().data_ptr<float>(), 
        (glm::vec3*)start_emissions.contiguous().data_ptr<float>(), 
        start_is_light_source.contiguous().data_ptr<bool>(), 
        out_idx_s.contiguous().data_ptr<int>(), 
        out_pdf_s.contiguous().data_ptr<float>(), 
        form_factor_cache.contiguous().data_ptr<int>(), 
        light_source_decay, inverse_falloff_max, min_decay
    );
    cudaDeviceSynchronize();
    return std::make_tuple(out_idx_s, out_pdf_s);
}

__global__ void estimateInClusterNextEvent(
    const unsigned long long seed, 
    const unsigned int active_sh_degree, 
    const unsigned int max_sh_degree, 
    const bool use_brdf_importance_sampling, 
    const bool clamp, 
    const unsigned int N, const unsigned int M, 
    const bool cluster_sampling, 
    const glm::vec3* __restrict__ means3D, 
    const glm::vec3* __restrict__ normals, 
    const glm::vec3* __restrict__ brdf_coeffs, 
    const bool* __restrict__ is_ls_s, 
    const int* __restrict__ stop_idx_s, 
    const int* __restrict__ next_cluster_idx_s, 
    const float* __restrict__ pdf_s, 
    const int* __restrict__ cluster_idx2offset, 
    const int* __restrict__ sorted_indices, 
    const int* __restrict__ sorted_cluster_idx, 
    const glm::vec3* __restrict__ sorted_means3D, 
    const glm::vec3* __restrict__ sorted_normals, 
    const glm::vec3* __restrict__ sorted_emissions, 
    const bool* __restrict__ sorted_is_ls_s, 
    int* __restrict__ out_idx_s, 
    float* __restrict__ out_pdf_s, 
    int* __restrict__ form_factor_cache, 
    const bool light_source_decay, 
    const float inverse_falloff_max, 
    const float min_decay
) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool done = idx >= N;

    const unsigned int SH_STRIDE = (max_sh_degree + 1) * (max_sh_degree + 1);
    int stop_idx = -1;
    register int next_cluster_idx = -1;
    register glm::vec3 stop_mean3D, stop_normal;
    bool is_ls;
    if ( !done ) {
        stop_idx = stop_idx_s[idx];
        next_cluster_idx = next_cluster_idx_s[idx];
        stop_mean3D = means3D[stop_idx];
        stop_normal = normals[stop_idx];
        is_ls = is_ls_s[stop_idx];
    }

    __shared__ int2 collected_indices[EXPANSION * BLOCK_SIZE];
    __shared__ glm::vec3 collected_means3D[EXPANSION * BLOCK_SIZE];
    __shared__ glm::vec3 collected_normals[EXPANSION * BLOCK_SIZE];
    __shared__ float collected_emissions[EXPANSION * BLOCK_SIZE];
    __shared__ bool collected_is_light_sources[EXPANSION * BLOCK_SIZE];

    __shared__ int from_cluster_idx;
    __shared__ int to_cluster_idx;

    if ( threadIdx.x == 0 ) {
        from_cluster_idx = INT_MAX;
        to_cluster_idx = -1;
    }

    __syncthreads();

    float unnormalized_probs[EXPANSION * BLOCK_SIZE] = {0};
    float accum_unnormalized_probs[EXPANSION * BLOCK_SIZE] = {0};
    int indices_buffer[EXPANSION * BLOCK_SIZE] = {0};
    int current_idx = -1;
    float current_unnormalized_prob = 0.0F;
    float current_equivalent_prob = 0.0F;

    if ( !done ) {
        atomicMin(&from_cluster_idx, next_cluster_idx);
        atomicMax(&to_cluster_idx, next_cluster_idx);
    }

    __syncthreads();

    if ( from_cluster_idx > to_cluster_idx ) return;
    
    const int from_sorted_idx = from_cluster_idx == 0 ? 0 : cluster_idx2offset[from_cluster_idx - 1];
    const int to_sorted_idx = cluster_idx2offset[to_cluster_idx];
    
    const int rounds = ((to_sorted_idx - from_sorted_idx) + (EXPANSION * BLOCK_SIZE) - 1) / (EXPANSION * BLOCK_SIZE);

    #pragma unroll 64
    for ( int r = 0; r < rounds; ++r ) {
        int num_done = __syncthreads_count(done);
        if ( num_done == BLOCK_SIZE )
            break;

        #pragma unroll 4
        for ( int f = 0; f < EXPANSION; ++f ) {
            const int sub_idx = (threadIdx.x * EXPANSION) + f;
            const int fetch_idx = r * (EXPANSION * BLOCK_SIZE) + sub_idx + from_sorted_idx;
            if ( fetch_idx < to_sorted_idx ) {
                collected_indices[sub_idx] = make_int2(sorted_indices[fetch_idx], sorted_cluster_idx[fetch_idx]);
                collected_means3D[sub_idx] = sorted_means3D[fetch_idx];
                collected_normals[sub_idx] = sorted_normals[fetch_idx];
                collected_emissions[sub_idx] = fabs(clampBaseProb(clamp? 0 : 3, computeOutRadianceFromSH(0, (sorted_emissions + fetch_idx * SH_STRIDE), glm::vec3(0.0F), clamp)));
                collected_is_light_sources[sub_idx] = sorted_is_ls_s[fetch_idx];
            }
        }
        __syncthreads();

        const int size = min(EXPANSION * BLOCK_SIZE, (to_sorted_idx - from_sorted_idx) - r * EXPANSION * BLOCK_SIZE);
        bool any_valid = false;
        float current_sum_unnormalized_prob = 0.0F;

        #pragma unroll 64
        for ( int i = 0; !done && i < size; ++i ) {
            const auto raw_idx = i + r * BLOCK_SIZE * EXPANSION + from_sorted_idx;
            const auto start_idx = collected_indices[i].x;
            if ( (!cluster_sampling && stop_idx == start_idx) || (next_cluster_idx != collected_indices[i].y) ) {
                unnormalized_probs[i] = 0.0F;
                indices_buffer[i] = start_idx;
                accum_unnormalized_probs[i] = ((i > 0)? accum_unnormalized_probs[i - 1] : 0.0F);
                continue;
            }
            const glm::vec3 start_mean3D = collected_means3D[i];
            const glm::vec3 start_normal = collected_normals[i];
            const glm::vec3 dir = glm::normalize(stop_mean3D - start_mean3D);
            const auto length = glm::length(stop_mean3D - start_mean3D);
            float fused_prob = 0.0F;
            const auto stop_angle_factor = (is_ls? 1.0F : -glm::dot(stop_normal, dir));
            const auto start_angle_factor = (collected_is_light_sources[i]? 1.0F : glm::dot(start_normal, dir));
            const auto inverse_square_law = (light_source_decay || (!is_ls && !collected_is_light_sources[i]))? min(inverse_falloff_max, 1.0F / (length * length + 1E-8F)) : 1.0F;
            if ( (cluster_sampling && start_angle_factor > 0.0F && stop_angle_factor > 0.0F && inverse_square_law > 0.0F) || (start_angle_factor > min_decay && stop_angle_factor > min_decay && inverse_square_law > min_decay) ) {
                // Rough Form Factor
                fused_prob = inverse_square_law * start_angle_factor * stop_angle_factor * collected_emissions[i];

                if ( fused_prob < min_decay ) fused_prob = 0.0F;
                else any_valid = true;
            }
            unnormalized_probs[i] = fused_prob;
            indices_buffer[i] = start_idx;
            accum_unnormalized_probs[i] = ((i > 0)? accum_unnormalized_probs[i - 1] : 0.0F) + fused_prob;
            current_sum_unnormalized_prob += fused_prob;
        }

        if ( !done && any_valid ) {
            // Sample from this batch
            auto res = sampleWithWeightedProbabilityBinarySearch(seed, idx, r * 2, accum_unnormalized_probs, current_sum_unnormalized_prob, size);
            // if (threadIdx.x == 0 && blockIdx.x == 0)
            //     printf("Selected Index: %d; Buffer: %d;\n", res, indices_buffer[res]);
            const float selected_unnorm_prob = unnormalized_probs[res];
            if ( current_idx < 0 ) {
                current_idx = indices_buffer[res];
                current_unnormalized_prob = selected_unnorm_prob;
                current_equivalent_prob = current_sum_unnormalized_prob;
            } else {
                int two_idx[] = {current_idx, indices_buffer[res]};
                float two_unnormalized_prob[] = { current_unnormalized_prob, selected_unnorm_prob };
                float two_accum_unnormalized_prob[] = { current_equivalent_prob, current_equivalent_prob + current_sum_unnormalized_prob };
                float two_sum_accum = current_equivalent_prob + current_sum_unnormalized_prob;
                auto res = sampleWithWeightedProbabilityBinarySearch(seed, idx, r * 2 + 1, two_accum_unnormalized_prob, two_sum_accum, 2);

                current_idx = two_idx[res];
                current_unnormalized_prob = two_unnormalized_prob[res];
                current_equivalent_prob = two_sum_accum;
            }
        }

        // if (threadIdx.x == 0 && blockIdx.x == 0)
        //     printf("Enter %d, %d; Size: %d; Any Valid: %d; Current Index: %d\n", collected_indices[0].x, collected_indices[0].y, size, (int)any_valid, current_idx);

        done |= collected_indices[0].y > next_cluster_idx;
    }

    if ( idx < N ) {
        out_idx_s[idx] = current_idx;
        out_pdf_s[idx] = pdf_s[idx] * current_unnormalized_prob / (current_equivalent_prob + 1E-8F);
    }
}

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
) {
    const unsigned int N = stop_idx_s.size(0);
    const unsigned int M = sorted_means3D.size(0);
    const bool cluster_sampling = means3D.size(0) != sorted_means3D.size(0);
    assert(sorted_indices.size(0) == M);
    assert(sorted_cluster_idx.size(0) == M);
    assert(sorted_normals.size(0) == M);
    assert(sorted_emissions.size(0) == M);
    assert(sorted_is_light_source.size(0) == M);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts = means3D.options().dtype(torch::kInt);
    auto out_idx_s = torch::full({N}, -1, int_opts);
    auto out_pdf_s = torch::full({N}, 0.0F, float_opts);
    estimateInClusterNextEvent <<< (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (
        seed, active_sh_degree, max_sh_degree, use_brdf_importance_sampling, clamp, N, M, cluster_sampling, 
        (glm::vec3*)means3D.contiguous().data_ptr<float>(), 
        (glm::vec3*)normals.contiguous().data_ptr<float>(), 
        (glm::vec3*)brdf_coeffs.contiguous().data_ptr<float>(), 
        is_light_source.contiguous().data_ptr<bool>(), 
        stop_idx_s.contiguous().data_ptr<int>(), 
        next_cluster_idx_s.contiguous().data_ptr<int>(), 
        pdf_s.contiguous().data_ptr<float>(), 
        cluster_idx2offset.contiguous().data_ptr<int>(), 
        sorted_indices.contiguous().data_ptr<int>(), 
        sorted_cluster_idx.contiguous().data_ptr<int>(), 
        (glm::vec3*)sorted_means3D.contiguous().data_ptr<float>(), 
        (glm::vec3*)sorted_normals.contiguous().data_ptr<float>(), 
        (glm::vec3*)sorted_emissions.contiguous().data_ptr<float>(), 
        sorted_is_light_source.contiguous().data_ptr<bool>(), 
        out_idx_s.contiguous().data_ptr<int>(), 
        out_pdf_s.contiguous().data_ptr<float>(), 
        form_factor_cache.contiguous().data_ptr<int>(), 
        light_source_decay, 
        inverse_falloff_max, 
        min_decay
    );
    cudaDeviceSynchronize();
    return std::make_tuple(out_idx_s, out_pdf_s);
}