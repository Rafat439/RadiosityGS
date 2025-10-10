#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <optix_types.h>

#include "next_event.h"
#include "optix_aux.h"
#include "optix_dev_ptx.h"
#include "solver.h"
#include "auxiliary.h"

// #define DEBUG

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
        auto err = cudaGetLastError(); \
        std::cerr << "\n[CUDA ERROR] '" << err << "' in " << __FILE__ << "\nLine " << __LINE__ << ": " << B; \
        throw std::runtime_error(B); \
    }

void __initialize_context() {
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
        // pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        
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
        pipelineCompileOptions.numAttributeValues = 8;
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

    const unsigned int flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
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
        reinterpret_cast<CUdeviceptr>(tempBuffer),
        bufferSizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(out_pointer),
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

    CHECK_CUDA( cuMemFree(reinterpret_cast<CUdeviceptr>(pointer)), "Releasing Handle Failed." );
    return;
}

__forceinline__ unsigned long long generateSeed(const unsigned long long max_seed) {
    return rand() % max_seed;
}

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
) {
    __initialize_context();

    srand(seed);

    const unsigned int num_elements = means3D.size(0);

    auto int_opts = means3D.options().dtype(torch::kInt);
    auto bool_opts = means3D.options().dtype(torch::kBool);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    auto dL_dmeans3D = torch::full({num_elements, 3}, 0.0F, float_opts);
    auto dL_dgeovalues = torch::full({num_elements, 1}, 0.0F, float_opts);
    auto dL_dscales = torch::full({num_elements, 2}, 0.0F, float_opts);
    auto dL_drots = torch::full({num_elements, 4}, 0.0F, float_opts);
    auto dL_dnormals = torch::full({num_elements, 3}, 0.0F, float_opts);
    auto dL_dnorm_factors = torch::full({num_elements, 1}, 0.0F, float_opts);

    /** Initialize Optix Ray Tracing */
    OptixTraversableHandle ohandle = (OptixTraversableHandle)(handle);

    OptixShaderBindingTable sbt = {};
    constexpr int sbt_record_size = (OPTIX_SBT_RECORD_HEADER_SIZE / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
    char record[sbt_record_size * 3];
    optixSbtRecordPackHeader(oprograms[0], record);
    optixSbtRecordPackHeader(oprograms[1], record + sbt_record_size);
    optixSbtRecordPackHeader(oprograms[2], record + sbt_record_size * 2);

    CUdeviceptr dp;
    cuMemAlloc(&dp, sizeof(LaunchParams));

    CUdeviceptr dsbt;
    cuMemAlloc(&dsbt, sbt_record_size * 3);
    cuMemcpyHtoD(dsbt, record, sbt_record_size * 3);
    
    sbt.raygenRecord = dsbt;
    sbt.missRecordBase = dsbt + sbt_record_size;
    sbt.hitgroupRecordBase = dsbt + sbt_record_size * 2;
    sbt.missRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes = sbt_record_size;
    sbt.missRecordCount = sbt.hitgroupRecordCount = 1;
    /** Done. */

    cudaStream_t stream_fused_calc;
    cudaStreamCreate(&stream_fused_calc);
    
    LaunchParams p = {};
    
    p.seed = generateSeed(max_seed);
    p.handle = handle;
    p.active_sh_degree = active_sh_degree;
    p.max_sh_degree = max_sh_degree;
    p.light_source_decay = light_source_decay;
    p.inverse_falloff_max = inverse_falloff_max;
    p.min_decay = min_decay;
    p.means3D = reinterpret_cast<CUdeviceptr>(means3D.contiguous().data_ptr<float>());
    p.geovalues = reinterpret_cast<CUdeviceptr>(geovalues.contiguous().data_ptr<float>());
    p.scales = reinterpret_cast<CUdeviceptr>(scales.contiguous().data_ptr<float>());
    p.rots = reinterpret_cast<CUdeviceptr>(rots.contiguous().data_ptr<float>());
    p.normals = reinterpret_cast<CUdeviceptr>(normals.contiguous().data_ptr<float>());
    p.norm_factors = reinterpret_cast<CUdeviceptr>(norm_factors.contiguous().data_ptr<float>());
    p.start_idx_s = reinterpret_cast<CUdeviceptr>(curr_idx_s.contiguous().data_ptr<int>());
    p.stop_idx_s = reinterpret_cast<CUdeviceptr>(next_idx_s.contiguous().data_ptr<int>());
    p.is_valid = reinterpret_cast<CUdeviceptr>(valid_s.contiguous().data_ptr<bool>());
    p.light_source_mask = reinterpret_cast<CUdeviceptr>(is_light_source.contiguous().data_ptr<bool>());
    p.form_factor_cache = reinterpret_cast<CUdeviceptr>(form_factor_cache.contiguous().data_ptr<int>());
    
    p.A = reinterpret_cast<CUdeviceptr>(A.contiguous().data_ptr<float>());
    p.clamp_A = clamp_A;
    p.B = reinterpret_cast<CUdeviceptr>(B.contiguous().data_ptr<float>());
    p.clamp_B = clamp_B;
    p.dL_dmeans3D = reinterpret_cast<CUdeviceptr>(dL_dmeans3D.contiguous().data_ptr<float>());
    p.dL_dgeovalues = reinterpret_cast<CUdeviceptr>(dL_dgeovalues.contiguous().data_ptr<float>());
    p.dL_dscales = reinterpret_cast<CUdeviceptr>(dL_dscales.contiguous().data_ptr<float>());
    p.dL_drots = reinterpret_cast<CUdeviceptr>(dL_drots.contiguous().data_ptr<float>());
    p.dL_dnormals = reinterpret_cast<CUdeviceptr>(dL_dnormals.contiguous().data_ptr<float>());
    p.dL_dnorm_factors = reinterpret_cast<CUdeviceptr>(dL_dnorm_factors.contiguous().data_ptr<float>());

    p.batch_size = curr_idx_s.size(0);
    p.num_walks = curr_idx_s.size(1);

    auto launch_num = curr_idx_s.size(0) * curr_idx_s.size(1);

    cuMemcpyHtoD(dp, &p, sizeof(LaunchParams));
    if ( launch_num <= 1024 ) {
        CHECK_OPTIX( optixLaunch(
            opipeline, 
            stream_fused_calc, dp, sizeof(LaunchParams), 
            &sbt, launch_num, 1, 1
        ), "Optix Launch Failed" );
    } else if ( launch_num <= 1024 * 1024 ) {
        CHECK_OPTIX( optixLaunch(
            opipeline, 
            stream_fused_calc, dp, sizeof(LaunchParams), 
            &sbt, 1024, (launch_num + 1024 - 1) / 1024, 1
        ), "Optix Launch Failed" );
    } else {
        CHECK_OPTIX( optixLaunch(
            opipeline, 
            stream_fused_calc, dp, sizeof(LaunchParams), 
            &sbt, 1024, 1024, (launch_num + (1024 * 1024) - 1) / (1024 * 1024)
        ), "Optix Launch Failed" );
    }
    CHECK_CUDA( cudaStreamSynchronize(stream_fused_calc), "Optix Failed." );

    cudaStreamDestroy(stream_fused_calc);

    cuMemFree(dp);
    cuMemFree(dsbt);

    return std::make_tuple(dL_dmeans3D, dL_dgeovalues, dL_dscales, dL_drots, dL_dnormals, dL_dnorm_factors);
}