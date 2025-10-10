#pragma once
#include "optix.h"

#define INDEX_2D(name, i, j) (*((name) + (i) * (2 * num_walks) + j))
#define INDEX_READY_2D(name, i, j) (*((name) + (i) * (num_walks) + j))

struct LaunchParams {
    // Backward Only
    // unsigned long long mode; // 0 - Forward || 1 - Backward 
    unsigned long long seed;
    unsigned int batch_size;
    unsigned int num_walks;
    unsigned int active_sh_degree;
    unsigned int max_sh_degree;
    bool light_source_decay;
    float inverse_falloff_max;
    float min_decay;
    OptixTraversableHandle handle;
    CUdeviceptr means3D;
    CUdeviceptr geovalues;
    CUdeviceptr scales;
    CUdeviceptr rots;
    CUdeviceptr norm_factors;
    CUdeviceptr normals;
    CUdeviceptr start_idx_s;
    CUdeviceptr stop_idx_s;
    CUdeviceptr is_valid;
    CUdeviceptr light_source_mask;
    CUdeviceptr form_factor_cache;

    CUdeviceptr A; // corresponds to `start_idx_s`
    bool clamp_A;
    CUdeviceptr B; // corresponds to `stop_idx_s`
    bool clamp_B;
    CUdeviceptr dL_dmeans3D;
    CUdeviceptr dL_dgeovalues;
    CUdeviceptr dL_dscales;
    CUdeviceptr dL_drots;
    CUdeviceptr dL_dnormals;
    CUdeviceptr dL_dnorm_factors;
};