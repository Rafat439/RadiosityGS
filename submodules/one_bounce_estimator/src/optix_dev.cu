#include <cstdint>
#include <cfloat>
#include <optix.h>
#include <optix_device.h>
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <cuda/std/utility>
#include "optix_aux.h"
#include "auxiliary.h"

extern "C" {
    __constant__ LaunchParams optixLaunchParams;

    struct OneBouncePayload {
        float     t_max;
        float     out;
        bool      is_valid;
        unsigned int start_idx;
        unsigned int stop_idx;
    };

    __forceinline__ __device__ glm::mat3 get_rotation_matrix_from_quaternion(const glm::vec4& q) {
        // Normalize quaternion to get valid rotation
        float s = rsqrtf(
            q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
        );
        float w = q.x * s;
        float x = q.y * s;
        float y = q.z * s;
        float z = q.w * s;

        // Compute rotation matrix from quaternion
        return glm::mat3(
            1.f - 2.f * (y * y + z * z),
            2.f * (x * y + w * z),
            2.f * (x * z - w * y),
            2.f * (x * y - w * z),
            1.f - 2.f * (x * x + z * z),
            2.f * (y * z + w * x),
            2.f * (x * z + w * y),
            2.f * (y * z - w * x),
            1.f - 2.f * (x * x + y * y)
        );
    }

    __forceinline__ __device__ float footprint_activation(float x) {
        if (x >= 4.2815f) return 0.99f;
        else if (x >= 0.f) return 1.f - exp(-0.03279f * powf(x, 3.4f));
        else return 0.f;
    }

    __global__ void __raygen__rg() {
        const uint3 launch_index = optixGetLaunchIndex();
        const uint3 launch_dims = optixGetLaunchDimensions();
        const auto idx = launch_index.x * (launch_dims.y * launch_dims.z) + launch_index.y * launch_dims.z + launch_index.z;
        // if ( idx == 0 || idx == optixLaunchParams.batch_size - 1 ) printf("(%d, %d, %d), (%d, %d, %d).\n", launch_index.x, launch_index.y, launch_index.z, launch_dims.x, launch_dims.y, launch_dims.z);
        if ( idx >= optixLaunchParams.batch_size ) return;
        bool enable = ((bool*)optixLaunchParams.stop_mask)[idx];
        if ( !enable ) return;
        const bool* is_light_source = ((bool*)optixLaunchParams.light_source_mask);
        const int* start_idx_s = ((int*)optixLaunchParams.start_idx_s);
        const int* stop_idx_s = ((int*)optixLaunchParams.stop_idx_s);
        const bool clamp = optixLaunchParams.clamp;
        const unsigned int active_sh_degree = optixLaunchParams.active_sh_degree;

        const float* geovalues = ((float*)optixLaunchParams.geovalues);
        const glm::vec3* means3D = ((glm::vec3*)optixLaunchParams.means3D);
        const glm::vec2* scales = ((glm::vec2*)optixLaunchParams.scales);
        const glm::vec4* rots = ((glm::vec4*)optixLaunchParams.rots);
        const glm::vec3* normals = ((glm::vec3*)optixLaunchParams.normals);
        const float* norm_factors = ((float*)optixLaunchParams.norm_factors);
        const glm::vec3* emissions = ((glm::vec3*)optixLaunchParams.emissions);
        const glm::vec3* brdf_coeffs = ((glm::vec3*)optixLaunchParams.brdf_coeffs);
        glm::vec3* out_form_factor = ((glm::vec3*)optixLaunchParams.out_form_factor);
        
        int start_idx = max(start_idx_s[idx], stop_idx_s[idx]);
        int stop_idx  = min(start_idx_s[idx], stop_idx_s[idx]);
        int emit_idx  = start_idx_s[idx];
        int recv_idx  = stop_idx_s[idx];

        if ( start_idx == stop_idx || start_idx < 0 || stop_idx < 0 ) return;
        
        const auto SH_STRIDE = ((optixLaunchParams.max_sh_degree + 1) * (optixLaunchParams.max_sh_degree + 1));

        const auto start_geovalue = geovalues[start_idx];
        const auto stop_geovalue = geovalues[stop_idx];

        if ( start_geovalue <= 0.5358F || stop_geovalue <= 0.5358F ) return;

        const auto start_means3D = means3D[start_idx];
        const auto stop_means3D = means3D[stop_idx];
        const auto start_normal = normals[start_idx];
        const auto stop_normal = normals[stop_idx];
        const auto start_scales = scales[start_idx];
        const auto stop_scales = scales[stop_idx];
        const auto start_rotation_matrix = get_rotation_matrix_from_quaternion(rots[start_idx]);
        const auto stop_rotation_matrix = get_rotation_matrix_from_quaternion(rots[stop_idx]);
        const auto start_is_light_source = is_light_source[start_idx];
        const auto stop_is_light_source = is_light_source[stop_idx];
        const auto recv_brdf_coeffs = brdf_coeffs + recv_idx * SH_STRIDE;
        
        // const auto& emit_is_light_source = (emit_idx == start_idx)? start_is_light_source : stop_is_light_source;
        const auto& emit_rotation_matrix = (emit_idx == start_idx)? start_rotation_matrix : stop_rotation_matrix;
        const auto& recv_rotation_matrix = (recv_idx == start_idx)? start_rotation_matrix : stop_rotation_matrix;

        const bool emit_is_ls = (emit_idx == start_idx)? start_is_light_source : stop_is_light_source;
        const bool recv_is_ls = (emit_idx == start_idx)? stop_is_light_source  : start_is_light_source;

        OneBouncePayload payload = {};
        unsigned int payload_addr_high = (reinterpret_cast<unsigned long long>(&payload)) >> 32;
        unsigned int payload_addr_low  = static_cast<uint32_t>((reinterpret_cast<unsigned long long>(&payload)));
        payload.start_idx = start_idx;
        payload.stop_idx = stop_idx;

        glm::vec3 ray_org = start_means3D;
        glm::vec3 ray_dir = glm::normalize(stop_means3D - start_means3D);
        payload.t_max = glm::length(stop_means3D - start_means3D);
        if ( payload.t_max < 1E-8F ) return;

        // const auto& emit_geovalue = (emit_idx == start_idx)? start_geovalue : stop_geovalue;
        // const auto& recv_geovalue = (emit_idx == start_idx)? stop_geovalue  : start_geovalue;

        // const auto& emit_scales = (emit_idx == start_idx)? start_scales : stop_scales;
        // const auto& recv_scales = (emit_idx == start_idx)? stop_scales  : start_scales;

        if ( optixLaunchParams.form_factor_cache == 0ull || read_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx) < 0 ) {
            float start_angle_factor = max(start_is_light_source ? 1.0F : glm::dot(start_normal, ray_dir), 0.0F);
            float stop_angle_factor = max(stop_is_light_source ? 1.0F : -glm::dot(stop_normal, ray_dir), 0.0F);
            float angle_factor = start_angle_factor * stop_angle_factor;
            if ( optixLaunchParams.light_source_decay || ( !start_is_light_source && !stop_is_light_source ) )
                angle_factor *= min(1.0F / (payload.t_max * payload.t_max), optixLaunchParams.inverse_falloff_max);
            
            payload.out = angle_factor;
            payload.is_valid = payload.out > optixLaunchParams.min_decay;
            if ( !payload.is_valid ) {
                write_to_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx, 0);
                return;
            }

            optixTrace(
                optixLaunchParams.handle, 
                make_float3(ray_org.x, ray_org.y, ray_org.z), 
                make_float3(ray_dir.x, ray_dir.y, ray_dir.z), 
                1E-8F, 
                payload.t_max, 
                0.0F, 
                OptixVisibilityMask(255), 
                OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 
                0, 1, 0, 
                payload_addr_high, 
                payload_addr_low
            );
            
            if ( !payload.is_valid ) {
                write_to_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx, 0);
                return;
            } else {
                write_to_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx, __float_as_int(payload.out));
            }
        } else {
            payload.out = __int_as_float(read_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx));
            payload.is_valid = read_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx) > 0;
        }

        if ( payload.is_valid ) {
            if ( clamp && !emit_is_ls )
                payload.out *= norm_factors[emit_idx];
            
            if ( !clamp && !recv_is_ls )
                payload.out *= norm_factors[recv_idx];
            
            const auto write_idx = optixLaunchParams.return_type ? idx : recv_idx;
            const glm::vec3 ray_dir_emit_frame = ray_dir * emit_rotation_matrix;
            const glm::vec3 ray_dir_recv_frame = ray_dir * recv_rotation_matrix;
            glm::vec3 c = computeOutRadianceFromSH(active_sh_degree, emissions + emit_idx * SH_STRIDE, ray_dir_emit_frame, clamp);
            if ( optixLaunchParams.return_type ) {
                computeSHResponse(c * payload.out, active_sh_degree, recv_brdf_coeffs, ray_dir_recv_frame, (out_form_factor + write_idx * SH_STRIDE));
            } else {
                glm::vec3 tmp[100] = { glm::vec3(0.0F) };
                computeSHResponse(c * payload.out, active_sh_degree, recv_brdf_coeffs, ray_dir_recv_frame, tmp);
                for ( unsigned int i = 0; i < SH_STRIDE; ++i )
                    atomicAdd(out_form_factor + write_idx * SH_STRIDE + i, tmp[i]);
            }
        }
    }

    __global__ void __anyhit__ch() {
        unsigned long long payload_addr_high = optixGetPayload_0();
        unsigned long long payload_addr_low  = optixGetPayload_1();
        unsigned long long payload_addr = (payload_addr_high << 32) + payload_addr_low;
        OneBouncePayload* payload = reinterpret_cast<OneBouncePayload*>(payload_addr);

        payload->out *= (1.f - footprint_activation(__uint_as_float(optixGetAttribute_0())));
        payload->is_valid = payload->out > optixLaunchParams.min_decay;
        
        if (payload->is_valid) optixIgnoreIntersection();
        else optixTerminateRay();
    }

    __global__ void __miss__far() {
        
    }

    __global__ void __intersection__aabb() {
        unsigned long long payload_addr_high = optixGetPayload_0();
        unsigned long long payload_addr_low  = optixGetPayload_1();
        unsigned long long payload_addr = (payload_addr_high << 32) + payload_addr_low;
        OneBouncePayload* payload = reinterpret_cast<OneBouncePayload*>(payload_addr);
        auto i = optixGetPrimitiveIndex();

        if ( i != payload->start_idx && i != payload->stop_idx) {
            // Fetch ray properties
            const float3 _rayOrigin = optixGetWorldRayOrigin();
            const float3 _rayDirection = optixGetWorldRayDirection();
            const glm::vec3 ray_org = glm::vec3(_rayOrigin.x, _rayOrigin.y, _rayOrigin.z);
            const glm::vec3 ray_dir = glm::vec3(_rayDirection.x, _rayDirection.y, _rayDirection.z);
            const auto t_min = 1e-8F;
            const auto t_max = payload->t_max;

            const auto _rot = get_rotation_matrix_from_quaternion(((glm::vec4*)optixLaunchParams.rots)[(i)]);
            const auto _mean3D = ((glm::vec3*)optixLaunchParams.means3D)[(i)];
            const auto _scale = ((glm::vec2*)optixLaunchParams.scales)[(i)];
            const auto _geovalue = ((float*)optixLaunchParams.geovalues)[(i)];

            const glm::vec3 local_ray_org = (ray_org - _mean3D) * _rot;
            const glm::vec3 local_ray_dir = ray_dir * _rot;
            const float t = -local_ray_org.z / local_ray_dir.z;

            if (abs(local_ray_dir.z) > 1E-8F && t < t_max && t > t_min) {
                const glm::vec3 intersect = local_ray_org + t * local_ray_dir;
                const float u = intersect.x / _scale.x;
                const float v = intersect.y / _scale.y;
                const float geovalue = exp(- 0.5f * (u * u + v * v)) * _geovalue;
                if ( geovalue > 0.5358F ) {
                    optixReportIntersection(t, 0, __float_as_uint(geovalue));
                }
            }
        }
    }
}