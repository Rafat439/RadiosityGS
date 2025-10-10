#pragma once
#include "optix.h"

struct LaunchParams {
    unsigned long long seed;
    unsigned int batch_size;
    unsigned int active_sh_degree;
    unsigned int max_sh_degree;
    bool clamp;
    bool return_type;
    bool light_source_decay;
    float inverse_falloff_max;
    float min_decay;
    OptixTraversableHandle handle;
    CUdeviceptr means3D;
    CUdeviceptr geovalues;
    CUdeviceptr scales;
    CUdeviceptr rots;
    CUdeviceptr normals;
    CUdeviceptr norm_factors;
    CUdeviceptr emissions;
    CUdeviceptr brdf_coeffs;
    CUdeviceptr start_idx_s;
    CUdeviceptr stop_idx_s;
    CUdeviceptr stop_mask;
    CUdeviceptr light_source_mask;
    CUdeviceptr out_form_factor;
    CUdeviceptr form_factor_cache;
};