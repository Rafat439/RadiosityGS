#include <cstdint>
#include <cfloat>
#include <optix.h>
#include <optix_device.h>
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include <cuda/std/utility>
#include "optix_aux.h"

extern "C" {
    __constant__ LaunchParams optixLaunchParams;

    struct FormFactorPayload {
        float     t_max;
        float     out;
        bool      is_valid;
        bool      mode;
        float     gradient;
        glm::vec3   dL_dray_org;
        glm::vec3   dL_dray_dir;
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

    __forceinline__ __device__ glm::vec4 quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
        float s = rsqrtf(
            quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
        );
        float w = quat.x * s;
        float x = quat.y * s;
        float y = quat.z * s;
        float z = quat.w * s;
    
        glm::vec4 v_quat;
        // v_R is COLUMN MAJOR
        // w element stored in x field
        v_quat.x =
            2.f * (
                    // v_quat.w = 2.f * (
                    x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                    z * (v_R[0][1] - v_R[1][0])
                );
        // x element in y field
        v_quat.y =
            2.f *
            (
                // v_quat.x = 2.f * (
                -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
                z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
            );
        // y element in z field
        v_quat.z =
            2.f *
            (
                // v_quat.y = 2.f * (
                x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
                z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
            );
        // z element in w field
        v_quat.w =
            2.f *
            (
                // v_quat.z = 2.f * (
                x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
                2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
            );
        return v_quat;
    }
    
    __forceinline__ __device__ float footprint_activation(float x) {
        if (x >= 4.2815f) return 0.99f;
        else if (x >= 0.f) return 1.f - exp(-0.03279f * powf(x, 3.4f));
        else return 0.f;
    }
    
    __forceinline__ __device__ float dfootprint_activation(float x) {
        if (x >= 4.2815f) return 0.f;
        else if (x >= 0.f) return 0.111486f * powf(x, 2.4f) * exp(-0.03279f * powf(x, 3.4f));
        else return 0.f;
    }

    __forceinline__ __device__ glm::vec3 dnormvdv(glm::vec3 v, glm::vec3 dv) {
        float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
        float invsum32 = 1.0f / (sqrt(sum2 * sum2 * sum2) + 1E-8F);
    
        glm::vec3 dnormvdv;
        dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
        dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
        dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
        return dnormvdv;
    }

    __global__ void __raygen__rg() {
        const uint3 launch_index = optixGetLaunchIndex();
        const uint3 launch_dims = optixGetLaunchDimensions();
        const auto idx = launch_index.x * (launch_dims.y * launch_dims.z) + launch_index.y * launch_dims.z + launch_index.z;
        if ( idx >= optixLaunchParams.batch_size ) return;

        const glm::vec3* start_coords_s = ((glm::vec3*)optixLaunchParams.start_coords);
        const glm::vec3* stop_coords_s = ((glm::vec3*)optixLaunchParams.stop_coords);
        float* out_visibility = ((float*)optixLaunchParams.out_visibility);
        const float* gradient = ((float*)optixLaunchParams.gradient);
        glm::vec3* dL_dstart_coords = ((glm::vec3*)optixLaunchParams.dL_dstart_coords);
        glm::vec3* dL_dstop_coords = ((glm::vec3*)optixLaunchParams.dL_dstop_coords);

        const glm::vec3 start_coords = start_coords_s[idx];
        const glm::vec3 stop_coords = stop_coords_s[idx];
        
        FormFactorPayload payload = {};
        unsigned int payload_addr_high = (reinterpret_cast<unsigned long long>(&payload)) >> 32;
        unsigned int payload_addr_low  = static_cast<uint32_t>((reinterpret_cast<unsigned long long>(&payload)));

        glm::vec3 ray_org = start_coords;
        glm::vec3 ray_dir = glm::normalize(stop_coords - start_coords);
        payload.t_max = glm::length(stop_coords - start_coords) - 1E-8F;
        payload.out = 1.F;
        payload.is_valid = true;
        if ( payload.t_max < 1E-8F ) return;

        if ( payload.mode == false ) {
            payload.mode = false;
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
    
            *(out_visibility + idx) = payload.out;
        } else {
            payload.mode = true;
            payload.out = *(out_visibility + idx);
            payload.gradient = *(gradient + idx);
            payload.dL_dray_org = glm::vec3(0.0F);
            payload.dL_dray_dir = glm::vec3(0.0F);
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
            *(dL_dstop_coords + idx) = dnormvdv(stop_coords - start_coords, payload.dL_dray_dir);
            *(dL_dstart_coords + idx) = payload.dL_dray_org - *(dL_dstop_coords + idx);
        }
    }

    __global__ void __anyhit__ch() {
        unsigned long long payload_addr_high = optixGetPayload_0();
        unsigned long long payload_addr_low  = optixGetPayload_1();
        unsigned long long payload_addr = (payload_addr_high << 32) + payload_addr_low;
        FormFactorPayload* payload = reinterpret_cast<FormFactorPayload*>(payload_addr);

        auto i = optixGetPrimitiveIndex();

        if ( payload->mode == false ) {
            // Forward Mode
            payload->out *= (1.f - footprint_activation(__uint_as_float(optixGetAttribute_2())));
            payload->is_valid = payload->out > optixLaunchParams.min_decay;

            if ( payload->is_valid ) optixIgnoreIntersection();
            else optixTerminateRay();
        } else {
            // Backward Mode

            const auto _quat = ((glm::vec4*)optixLaunchParams.rots)[(i)];
            const auto _rot = get_rotation_matrix_from_quaternion(_quat);
            const auto _mean3D = ((glm::vec3*)optixLaunchParams.means3D)[(i)];
            const auto _scale = ((glm::vec2*)optixLaunchParams.scales)[(i)];
            const auto _geovalue = ((float*)optixLaunchParams.geovalues)[(i)];

            const float t = __uint_as_float(optixGetAttribute_0());
            const float G = __uint_as_float(optixGetAttribute_1());
            const float geovalue = __uint_as_float(optixGetAttribute_2());
            const float u = __uint_as_float(optixGetAttribute_3());
            const float v = __uint_as_float(optixGetAttribute_4());
            const float alpha = (1.f - footprint_activation(geovalue));
            
            const float3 _rayOrigin = optixGetWorldRayOrigin();
            const float3 _rayDirection = optixGetWorldRayDirection();
            const glm::vec3 ray_org = glm::vec3(_rayOrigin.x, _rayOrigin.y, _rayOrigin.z);
            const glm::vec3 ray_dir = glm::vec3(_rayDirection.x, _rayDirection.y, _rayDirection.z);

            const glm::vec3 local_ray_dir = glm::vec3(
                __uint_as_float(optixGetAttribute_5()), __uint_as_float(optixGetAttribute_6()), __uint_as_float(optixGetAttribute_7())
            );

            const float dL_dalpha = payload->gradient * payload->out / alpha;
            const float dL_dgeovalue = - dL_dalpha * dfootprint_activation(geovalue);
            
            const float dL_dG = dL_dgeovalue * _geovalue;
            const glm::vec2 uv = glm::vec2(u, v);
            const glm::vec2 dL_duv = dL_dG * G * (-uv);
            const glm::vec2 inv_scale = 1.0F / _scale;
            const glm::vec2 dL_dscale = dL_duv * uv * (-inv_scale);

            const glm::vec3 dL_dintersect = glm::vec3(dL_duv * inv_scale, 0.0F);
            const float dL_dt = glm::dot(dL_dintersect, local_ray_dir);
            glm::vec3 dL_dlocal_org = dL_dintersect;
            glm::vec3 dL_dlocal_dir = t * dL_dintersect;
            
            dL_dlocal_org.z += dL_dt * (-1.0F / local_ray_dir.z);
            dL_dlocal_dir.z += dL_dt * t * (-1.0F / local_ray_dir.z);
            const glm::vec3 dL_ddist = _rot * dL_dlocal_org;

            const glm::vec4 dL_dq = quat_to_rotmat_vjp(_quat, glm::outerProduct(ray_org - _mean3D, dL_dlocal_org) + glm::outerProduct(ray_dir, dL_dlocal_dir));

            payload->dL_dray_org += dL_ddist;
            payload->dL_dray_dir += _rot * dL_dlocal_dir;

            atomicAdd(&(((float*)optixLaunchParams.dL_dgeovalues)[i]), dL_dgeovalue * G);

            atomicAdd(&(((float2*)optixLaunchParams.dL_dscales + (i))->x), dL_dscale.x);
            atomicAdd(&(((float2*)optixLaunchParams.dL_dscales + (i))->y), dL_dscale.y);

            atomicAdd(&(((float3*)optixLaunchParams.dL_dmeans3D + (i))->x), -dL_ddist.x);
            atomicAdd(&(((float3*)optixLaunchParams.dL_dmeans3D + (i))->y), -dL_ddist.y);
            atomicAdd(&(((float3*)optixLaunchParams.dL_dmeans3D + (i))->z), -dL_ddist.z);

            atomicAdd(&(((float4*)optixLaunchParams.dL_drots + (i))->x), dL_dq.x);
            atomicAdd(&(((float4*)optixLaunchParams.dL_drots + (i))->y), dL_dq.y);
            atomicAdd(&(((float4*)optixLaunchParams.dL_drots + (i))->z), dL_dq.z);
            atomicAdd(&(((float4*)optixLaunchParams.dL_drots + (i))->w), dL_dq.w);

            optixIgnoreIntersection();
        }
    }

    __global__ void __miss__far() {
        
    }

    __global__ void __intersection__aabb() {
        unsigned long long payload_addr_high = optixGetPayload_0();
        unsigned long long payload_addr_low  = optixGetPayload_1();
        unsigned long long payload_addr = (payload_addr_high << 32) + payload_addr_low;
        FormFactorPayload* payload = reinterpret_cast<FormFactorPayload*>(payload_addr);
        auto i = optixGetPrimitiveIndex();

        // Fetch ray properties
        const float3 _rayOrigin = optixGetWorldRayOrigin();
        const float3 _rayDirection = optixGetWorldRayDirection();
        const glm::vec3 ray_org = glm::vec3(_rayOrigin.x, _rayOrigin.y, _rayOrigin.z);
        const glm::vec3 ray_dir = glm::vec3(_rayDirection.x, _rayDirection.y, _rayDirection.z);
        const auto t_min = 1E-8F;
        const auto t_max = payload->t_max - 1E-8F;

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
            const float G = exp(- 0.5f * (u * u + v * v));
            const float geovalue = G * _geovalue;
            if ( geovalue > 0.5358F ) {
                optixReportIntersection(t, 0, __float_as_uint(t), __float_as_uint(G), __float_as_uint(geovalue), __float_as_uint(u), __float_as_uint(v), __float_as_uint(local_ray_dir.x), __float_as_uint(local_ray_dir.y), __float_as_uint(local_ray_dir.z));
            }
        }
    }
}