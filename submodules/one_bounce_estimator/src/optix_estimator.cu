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
#include "optix_estimator.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <optix_types.h>

#include "auxiliary.h"
#include "optix_aux.h"
#include "optix_dev_ptx.h"

static OptixDeviceContext ocontext = nullptr;
static OptixModule omodule = nullptr;
static OptixProgramGroup oprograms[3] = { nullptr, nullptr, nullptr };
static OptixPipeline opipeline = nullptr;

#define CHECK_OPTIX(A, B) \
    if ( A != OPTIX_SUCCESS ) { \
        std::cerr << "\n[OPTIX ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << B; \
        throw std::runtime_error(B); \
    }

#define CHECK_CUDA(A, B) \
    if ( A != CUDA_SUCCESS ) { \
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << ", " << B; \
        throw std::runtime_error(B); \
    }

__forceinline__ void __initialize_context() {
    if ( !ocontext ) {
        // std::cout << "Initializing Optix... The first time loading will take some time." << std::endl;
        OptixDeviceContextOptions options = {  };
        // options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        CHECK_OPTIX( optixInit(), "Initialization Failed." );
        CHECK_OPTIX( optixDeviceContextCreate(nullptr, &options, &ocontext), "Context Creation Failed." );
    }
    OptixModuleCompileOptions moduleCompileOptions = {  };
    OptixPipelineCompileOptions pipelineCompileOptions = {  };
    OptixPipelineLinkOptions pipelineLinkOptions = {  };
    if ( !omodule ) {
        moduleCompileOptions.maxRegisterCount = 128;
        // moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        // pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
        
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

        // optixDeviceContextSetLogCallback(
        //     ocontext,
        //     [](unsigned int level, const char* tag, const char* message, void*) {
        //         printf("[OptiX][%d][%s] %s\n", level, tag, message);
        //     },
        //     nullptr,
        //     4  // Maximum verbosity level
        // );

        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 1;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        pipelineLinkOptions.maxTraceDepth = 1;

        char log[2048] = "OptiX Create Module Failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);

        CHECK_OPTIX( optixModuleCreate(ocontext, &moduleCompileOptions, &pipelineCompileOptions, (const char*)optix_dev_ptx, optix_dev_ptx_len, log + strlen(log), &sizeof_log, &omodule), log );
    }

    if ( !oprograms[0] ) {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = omodule;
        pgDesc.raygen.entryFunctionName = "__raygen__rg";

        char log[2048] = "Optix Create RayGen Program Failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        CHECK_OPTIX( optixProgramGroupCreate(ocontext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &oprograms[0]), log );
    }

    if ( !oprograms[1] ) {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.raygen.module = omodule;
        pgDesc.raygen.entryFunctionName = "__miss__far";

        char log[2048] = "Optix Create Miss Program Failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        CHECK_OPTIX( optixProgramGroupCreate(ocontext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &oprograms[1]), log );
    }

    if ( !oprograms[2] ) {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = nullptr;
        pgDesc.hitgroup.moduleAH = omodule;
        pgDesc.hitgroup.moduleIS = omodule;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__ch";
        pgDesc.hitgroup.entryFunctionNameIS = "__intersection__aabb";

        char log[2048] = "Optix Create Hit Program Failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        CHECK_OPTIX( optixProgramGroupCreate(ocontext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &oprograms[2]), log );
    }

    if ( !opipeline ) {
        char log[2048] = "Optix Create Pipeline Failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        CHECK_OPTIX( optixPipelineCreate(ocontext, &pipelineCompileOptions, &pipelineLinkOptions, oprograms, 3, log, &sizeof_log, &opipeline), log );

        CHECK_OPTIX( optixPipelineSetStackSize(opipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1), "Optix Pipeline Set Stack Size Failed." );
    }
}

std::tuple<unsigned long long, unsigned long long> OptixBuildBVH(
    const torch::Tensor& aabbs
) {
    __initialize_context();
    OptixTraversableHandle out_handle;
    CUdeviceptr out_pointer;

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    
    const CUdeviceptr bboxes = reinterpret_cast<const CUdeviceptr>(aabbs.contiguous().data_ptr<float>());
    buildInput.customPrimitiveArray.aabbBuffers = &bboxes;
    buildInput.customPrimitiveArray.numPrimitives = aabbs.size(0);
    buildInput.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);

    const unsigned int flags[] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
    buildInput.customPrimitiveArray.flags = flags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;
    CHECK_OPTIX( optixAccelComputeMemoryUsage(ocontext,
        &buildOptions,
        &buildInput,
        1,  // num_build_inputs
        &bufferSizes
    ), "Computing Acceleration Structure Memory Usage Failed." );

    CUdeviceptr tempBuffer;
    CHECK_CUDA( cuMemAlloc(&tempBuffer, bufferSizes.tempSizeInBytes), "Temporary Buffer Initialization Failed." );
    CHECK_CUDA( cuMemAlloc(&out_pointer, bufferSizes.outputSizeInBytes), "BVH Memory Allocation Failed." );
    
    CHECK_OPTIX( optixAccelBuild(ocontext,
        0,
        &buildOptions,
        &buildInput,
        1,
        (CUdeviceptr)tempBuffer,
        bufferSizes.tempSizeInBytes,
        (CUdeviceptr)out_pointer,
        bufferSizes.outputSizeInBytes,
        &out_handle,
        nullptr, 0
    ), "BVH Acceleration Structure Building Failed.");

    CHECK_CUDA( cuCtxSynchronize(), "Context Synchronization Failed." );
    CHECK_CUDA( cuMemFree(tempBuffer), "Cleaning Up Temporary Buffer Failed." );

    return std::make_tuple(out_handle, out_pointer);
}

void OptixReleaseBVH(
    unsigned long long handle, 
    unsigned long long pointer
) {
    __initialize_context();

    CHECK_CUDA( cuMemFree((CUdeviceptr)pointer), "Releasing Handle Failed." );
    return;
}

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
) {
    __initialize_context();
    const auto SH_STRIDE = ((max_sh_degree + 1) * (max_sh_degree + 1));

    const auto n_rays = start_idx_s.size(0);
    const auto n_elements = means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto out_form_factor = torch::full({return_type? n_rays : n_elements, SH_STRIDE, 3}, 0.0f, float_opts);

    OptixTraversableHandle ohandle = (OptixTraversableHandle)(handle);

    OptixShaderBindingTable sbt = {};
    constexpr int sbt_record_size = (OPTIX_SBT_RECORD_HEADER_SIZE / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
    char record[sbt_record_size * 3];
    optixSbtRecordPackHeader(oprograms[0], record);
    optixSbtRecordPackHeader(oprograms[1], record + sbt_record_size);
    optixSbtRecordPackHeader(oprograms[2], record + sbt_record_size * 2);

    CUdeviceptr dsbt;
    CHECK_CUDA( cuMemAlloc(&dsbt, sbt_record_size * 3), "Memory Allocation Failed." );
    CHECK_CUDA( cuMemcpyHtoD(dsbt, record, sbt_record_size * 3), "Memory Copy Failed." );
    
    sbt.raygenRecord = dsbt;
    sbt.missRecordBase = dsbt + sbt_record_size;
    sbt.hitgroupRecordBase = dsbt + sbt_record_size * 2;
    sbt.missRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes = sbt_record_size;
    sbt.missRecordCount = sbt.hitgroupRecordCount = 1;

    LaunchParams p = {};
    p.seed = seed;
    p.handle = ohandle;
    p.active_sh_degree = active_sh_degree;
    p.max_sh_degree = max_sh_degree;
    p.clamp = clamp;
    p.batch_size = n_rays;
    p.return_type = return_type;
    p.light_source_decay = light_source_decay;
    p.inverse_falloff_max = inverse_falloff_max;
    p.min_decay = min_decay;
    p.means3D = (CUdeviceptr)(means3D.contiguous().data_ptr<float>());
    p.geovalues = (CUdeviceptr)(geovalues.contiguous().data_ptr<float>());
    p.scales = (CUdeviceptr)(scales.contiguous().data_ptr<float>());
    p.rots = (CUdeviceptr)(rots.contiguous().data_ptr<float>());
    p.normals = (CUdeviceptr)(normals.contiguous().data_ptr<float>());
    p.norm_factors = (CUdeviceptr)(norm_factors.contiguous().data_ptr<float>());
    p.emissions = (CUdeviceptr)(emissions.contiguous().data_ptr<float>());
    p.brdf_coeffs = (CUdeviceptr)(brdf_coeffs.contiguous().data_ptr<float>());
    p.start_idx_s = (CUdeviceptr)(start_idx_s.contiguous().data_ptr<int>());
    p.stop_idx_s = (CUdeviceptr)(stop_idx_s.contiguous().data_ptr<int>());
    p.stop_mask = (CUdeviceptr)(stop_mask.contiguous().data_ptr<bool>());
    p.light_source_mask = (CUdeviceptr)(light_source_mask.contiguous().data_ptr<bool>());
    p.out_form_factor = (CUdeviceptr)(out_form_factor.contiguous().data_ptr<float>());
    p.form_factor_cache = (CUdeviceptr)(form_factor_cache.contiguous().data_ptr<int>());
    
    CUdeviceptr dp;
    CHECK_CUDA( cuMemAlloc(&dp, sizeof(LaunchParams)), "Memory Allocation Failed." );
    CHECK_CUDA( cuMemcpyHtoD(dp, &p, sizeof(LaunchParams)), "Memory Copy Failed." );

    if ( n_rays <= 1024 ) {
        CHECK_OPTIX( optixLaunch(opipeline, 0, (CUdeviceptr)dp, sizeof(LaunchParams), &sbt, n_rays, 1, 1), "Optix Launch Failed." );
    } else if ( n_rays <= 1024 * 1024 ) {
        CHECK_OPTIX( optixLaunch(opipeline, 0, (CUdeviceptr)dp, sizeof(LaunchParams), &sbt, 1024, (n_rays + 1024 - 1) / 1024, 1), "Optix Launch Failed." );
    } else {
        CHECK_OPTIX( optixLaunch(opipeline, 0, (CUdeviceptr)dp, sizeof(LaunchParams), &sbt, 1024, 1024, (n_rays + (1024 * 1024) - 1) / (1024 * 1024)), "Optix Launch Failed." );
    }
    CHECK_CUDA( cuCtxSynchronize(), "Context Synchronization Failed." );
    CHECK_CUDA( cuMemFree(dp), "Memory Release Failed." );
    CHECK_CUDA( cuMemFree(dsbt), "Memory Release Failed." );

    return out_form_factor;
}