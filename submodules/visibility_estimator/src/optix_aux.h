#pragma once
#include "optix.h"

struct LaunchParams {
    unsigned int batch_size;
    float min_decay;
    bool mode; // 0 - forward | 1 - backward
    OptixTraversableHandle handle;
    CUdeviceptr means3D;
    CUdeviceptr geovalues;
    CUdeviceptr scales;
    CUdeviceptr rots;
    CUdeviceptr start_coords;
    CUdeviceptr stop_coords;
    CUdeviceptr out_visibility;

    CUdeviceptr gradient;
    CUdeviceptr dL_dmeans3D;
    CUdeviceptr dL_dgeovalues;
    CUdeviceptr dL_dscales;
    CUdeviceptr dL_drots;
    CUdeviceptr dL_dstart_coords;
    CUdeviceptr dL_dstop_coords;
};