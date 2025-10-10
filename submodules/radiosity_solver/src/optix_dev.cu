#include <cstdint>
#include <cfloat>
#include "optix.h"
#include <optix_device.h>
#include <glm/glm.hpp>
#include <curand_kernel.h>
#include "optix_aux.h"
#include "auxiliary.h"

extern "C" {
    __constant__ LaunchParams optixLaunchParams;

    struct OneBouncePayload {
        unsigned int mode;

        float       out;
        float       t_max;
        bool        is_valid;

        unsigned int start_idx;
        unsigned int stop_idx;

        float       gradient;
        glm::vec3   dL_dray_org;
        glm::vec3   dL_dray_dir;
    };

    __global__ void __raygen__rg() {
        const uint3 launch_index = optixGetLaunchIndex();
        const uint3 launch_dims = optixGetLaunchDimensions();
        const auto idx = launch_index.x * (launch_dims.y * launch_dims.z) + launch_index.y * launch_dims.z + launch_index.z;
        const auto batch_size = optixLaunchParams.batch_size;
        const auto num_walks = optixLaunchParams.num_walks;
        const auto batch_idx = idx / num_walks;
        const auto walk_idx = idx % num_walks;
        // if ( idx == 0 || idx == (launch_dims.x * launch_dims.y * launch_dims.z - 1) ) printf("(%d,%d,%d),(%d,%d,%d);(%d,%d),(%d,%d)\n", launch_index.x, launch_index.y, launch_index.z, launch_dims.x, launch_dims.y, launch_dims.z, batch_idx, walk_idx, batch_size, num_walks);
        
        if ( batch_idx >= batch_size ) return;
        bool enable = INDEX_READY_2D((bool*)optixLaunchParams.is_valid, batch_idx, walk_idx);
        if ( !enable ) return;

        const bool* is_light_source = ((bool*)optixLaunchParams.light_source_mask);
        const int* start_idx_s = ((int*)optixLaunchParams.start_idx_s);
        const int* stop_idx_s = ((int*)optixLaunchParams.stop_idx_s);
        const glm::vec3* means3D = ((glm::vec3*)optixLaunchParams.means3D);
        const float* geovalues = ((float*)optixLaunchParams.geovalues);
        const glm::vec2* scales = ((glm::vec2*)optixLaunchParams.scales);
        const glm::vec4* rots = ((glm::vec4*)optixLaunchParams.rots);
        const glm::vec3* normals = ((glm::vec3*)optixLaunchParams.normals);
        const float* norm_factors = ((float*)optixLaunchParams.norm_factors);

        int A_idx = INDEX_READY_2D(start_idx_s, batch_idx, walk_idx);
        int B_idx = INDEX_READY_2D(stop_idx_s, batch_idx, walk_idx);

        if ( A_idx == B_idx || A_idx < 0 || B_idx < 0 ) return;

        const auto SH_STRIDE = (optixLaunchParams.max_sh_degree + 1) * (optixLaunchParams.max_sh_degree + 1);
        const glm::vec3* A = ((glm::vec3*)optixLaunchParams.A) + A_idx * SH_STRIDE;
        const glm::vec3* B = ((glm::vec3*)optixLaunchParams.B) + B_idx * SH_STRIDE;

        unsigned int start_idx = max(A_idx, B_idx);
        unsigned int stop_idx  = min(A_idx, B_idx);
        
        const auto start_means3D = means3D[start_idx];
        const auto stop_means3D = means3D[stop_idx];
        const auto start_geovalue = geovalues[start_idx];
        const auto stop_geovalue = geovalues[stop_idx];

        if ( start_geovalue <= 0.5358F || stop_geovalue <= 0.5358F ) return;

        const auto start_quats = rots[start_idx];
        const auto stop_quats = rots[stop_idx];
        const auto start_normal = normals[start_idx];
        const auto stop_normal = normals[stop_idx];
        const auto start_scales = scales[start_idx];
        const auto stop_scales = scales[stop_idx];

        const auto start_rotation_matrix = get_rotation_matrix_from_quaternion(start_quats);
        const auto stop_rotation_matrix = get_rotation_matrix_from_quaternion(stop_quats);
        const auto start_is_light_source = is_light_source[start_idx];
        const auto stop_is_light_source = is_light_source[stop_idx];
        const auto start_clamp = (A_idx == start_idx)? optixLaunchParams.clamp_A : optixLaunchParams.clamp_B;
        const auto stop_clamp = (B_idx == start_idx)? optixLaunchParams.clamp_A : optixLaunchParams.clamp_B;
        
        const auto& A_quats = (A_idx == start_idx)? start_quats : stop_quats;
        const auto& A_rotation_matrix = (A_idx == start_idx)? start_rotation_matrix : stop_rotation_matrix;
        const auto& B_rotation_matrix = (B_idx == start_idx)? start_rotation_matrix : stop_rotation_matrix;

        glm::vec3* dL_dmeans3D = ((glm::vec3*)optixLaunchParams.dL_dmeans3D);
        glm::vec3* dL_dnormals = ((glm::vec3*)optixLaunchParams.dL_dnormals);
        glm::vec2* dL_dscales  = ((glm::vec2*)optixLaunchParams.dL_dscales);
        glm::vec4* dL_drots    = ((glm::vec4*)optixLaunchParams.dL_drots);
        float* dL_dgeovalues = ((float*)optixLaunchParams.dL_dgeovalues);
        float* dL_dnorm_factors = ((float*)optixLaunchParams.dL_dnorm_factors);

        unsigned int forward_mode = 0;
        unsigned int backward_mode = 1;

        glm::vec3 dL_dmeans3D_start_idx = glm::vec3(0.0F);
        glm::vec3 dL_dmeans3D_stop_idx  = glm::vec3(0.0F);

        glm::vec3 dL_dnormals_start_idx = glm::vec3(0.0F);
        glm::vec3 dL_dnormals_stop_idx  = glm::vec3(0.0F);

        glm::vec3 dL_dscales_start_idx  = glm::vec3(0.0F);
        glm::vec3 dL_dscales_stop_idx   = glm::vec3(0.0F);

        glm::vec4 dL_drots_start_idx    = glm::vec4(0.0F);
        glm::vec4 dL_drots_stop_idx     = glm::vec4(0.0F);

        float dL_dgeovalues_start_idx = 0.0F;
        float dL_dgeovalues_stop_idx = 0.0F;

        register OneBouncePayload payload = {};
        unsigned int payload_addr_high = (reinterpret_cast<unsigned long long>(&payload)) >> 32;
        unsigned int payload_addr_low  = static_cast<uint32_t>((reinterpret_cast<unsigned long long>(&payload)));

        payload.start_idx = start_idx;
        payload.stop_idx = stop_idx;

        glm::vec3 ray_org = start_means3D;
        glm::vec3 ray_dir = glm::normalize(stop_means3D - start_means3D);
        payload.dL_dray_org = glm::vec3(0.0F);
        payload.dL_dray_dir = glm::vec3(0.0F);
        payload.t_max = glm::length(stop_means3D - start_means3D);
        if ( payload.t_max < 1E-8F ) return;

        float angle_factor_start = max(start_is_light_source ? 1.0F : glm::dot(start_normal, ray_dir), 0.0F);
        float angle_factor_stop = max(stop_is_light_source ? 1.0F : -glm::dot(stop_normal, ray_dir), 0.0F);

        if ( optixLaunchParams.form_factor_cache == 0ull || read_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx) < 0 ) {
            float angle_factor = angle_factor_start * angle_factor_stop;
            if ( optixLaunchParams.light_source_decay || (!start_is_light_source && !stop_is_light_source) )
                angle_factor *= min(optixLaunchParams.inverse_falloff_max, 1.0F / (payload.t_max * payload.t_max));
            
            payload.out = angle_factor;
            payload.is_valid = payload.out > optixLaunchParams.min_decay;
            if ( !payload.is_valid ) {
                write_to_cached_form_factor((int*)optixLaunchParams.form_factor_cache, start_idx, stop_idx, 0);
                return;
            }

            payload.mode = forward_mode;
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
            if ( !payload.is_valid ) return;
        }

        if ( start_clamp && !start_is_light_source )
            payload.out *= norm_factors[start_idx];
        if ( stop_clamp && !stop_is_light_source )
            payload.out *= norm_factors[stop_idx];

        const glm::vec3 ray_dir_A = ray_dir * A_rotation_matrix;
        const glm::vec3 ray_dir_B = ray_dir * B_rotation_matrix;
        const glm::vec3 _B = computeOutRadianceFromSH(optixLaunchParams.active_sh_degree, B, ray_dir_B, optixLaunchParams.clamp_B);
        const glm::vec3 _A = computeOutRadianceFromSH(optixLaunchParams.active_sh_degree, A, ray_dir_A, optixLaunchParams.clamp_A);
        float dot = glm::dot(_A, _B);
            
        // dL / dtransmittance
        payload.gradient = (1.0F / (float)(batch_size)) * dot;
        
        if ( start_clamp && !start_is_light_source ) {
            const auto dL_dstart_o = payload.gradient * payload.out / norm_factors[start_idx];
            atomicAdd(&(dL_dnorm_factors[start_idx]), dL_dstart_o);
        }
        
        if ( stop_clamp && !stop_is_light_source ) {
            const auto dL_dstop_o  = payload.gradient * payload.out / norm_factors[stop_idx];
            atomicAdd(&(dL_dnorm_factors[stop_idx]), dL_dstop_o);
        }

        payload.mode = backward_mode;
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
        
        const float dL_dangle_factor = payload.gradient * payload.out / (angle_factor_start * angle_factor_stop);
        const float dL_dangle_factor_start = start_is_light_source ? 0.0F : dL_dangle_factor * angle_factor_stop;
        const float dL_dangle_factor_stop  = stop_is_light_source ? 0.0F : dL_dangle_factor * angle_factor_start;
        payload.dL_dray_dir += dL_dangle_factor_start * start_normal - dL_dangle_factor_stop * stop_normal;

        const float dL_dinverse_square_length = payload.gradient * payload.out * max(1.0F / optixLaunchParams.inverse_falloff_max, payload.t_max * payload.t_max);
        float dL_dlength = 0.0F;
        if ( (optixLaunchParams.light_source_decay || (!start_is_light_source && !stop_is_light_source)) && payload.t_max * payload.t_max * optixLaunchParams.inverse_falloff_max > 1.0F )
            dL_dlength = dL_dinverse_square_length * (-2.0F / (payload.t_max * payload.t_max * payload.t_max));

        const auto dL_dstop_pose = dnormvdv(stop_means3D - start_means3D, payload.dL_dray_dir) + dL_dlength * ray_dir;
        const auto dL_dstart_pose = payload.dL_dray_org - dL_dstop_pose;

        dL_dmeans3D_start_idx += dL_dstart_pose;
        dL_dmeans3D_stop_idx  += dL_dstop_pose;

        dL_dnormals_start_idx += +dL_dangle_factor_start * ray_dir;
        dL_dnormals_stop_idx  += -dL_dangle_factor_stop  * ray_dir;

        if ( optixLaunchParams.active_sh_degree > 0 ) {
            const glm::vec3 dL_dray_dir_A = grad_sh_2_grad_dir(optixLaunchParams.active_sh_degree, (1.0F / (float)(batch_size)) * payload.out * _B, A, ray_dir_A);

            const auto dL_dstop_pose = dnormvdv(stop_means3D - start_means3D, A_rotation_matrix * dL_dray_dir_A);
            const auto dL_dstart_pose = - dL_dstop_pose;

            dL_dmeans3D_start_idx += dL_dstart_pose;
            dL_dmeans3D_stop_idx  += dL_dstop_pose;

            (start_idx == A_idx ? dL_drots_start_idx : dL_drots_stop_idx) += quat_to_rotmat_vjp(A_quats, glm::outerProduct(ray_dir, dL_dray_dir_A));
        }

        atomicAdd(&(dL_dmeans3D[start_idx].x), dL_dmeans3D_start_idx.x);
        atomicAdd(&(dL_dmeans3D[start_idx].y), dL_dmeans3D_start_idx.y);
        atomicAdd(&(dL_dmeans3D[start_idx].z), dL_dmeans3D_start_idx.z);

        atomicAdd(&(dL_dmeans3D[stop_idx].x), dL_dmeans3D_stop_idx.x);
        atomicAdd(&(dL_dmeans3D[stop_idx].y), dL_dmeans3D_stop_idx.y);
        atomicAdd(&(dL_dmeans3D[stop_idx].z), dL_dmeans3D_stop_idx.z);

        atomicAdd(&(dL_dnormals[start_idx].x), dL_dnormals_start_idx.x);
        atomicAdd(&(dL_dnormals[start_idx].y), dL_dnormals_start_idx.y);
        atomicAdd(&(dL_dnormals[start_idx].z), dL_dnormals_start_idx.z);

        atomicAdd(&(dL_dnormals[stop_idx].x), dL_dnormals_stop_idx.x);
        atomicAdd(&(dL_dnormals[stop_idx].y), dL_dnormals_stop_idx.y);
        atomicAdd(&(dL_dnormals[stop_idx].z), dL_dnormals_stop_idx.z);

        atomicAdd(&(dL_dscales[start_idx].x), dL_dscales_start_idx.x);
        atomicAdd(&(dL_dscales[start_idx].y), dL_dscales_start_idx.y);

        atomicAdd(&(dL_dscales[stop_idx].x), dL_dscales_stop_idx.x);
        atomicAdd(&(dL_dscales[stop_idx].y), dL_dscales_stop_idx.y);

        atomicAdd(&(dL_drots[start_idx].x), dL_drots_start_idx.x);
        atomicAdd(&(dL_drots[start_idx].y), dL_drots_start_idx.y);
        atomicAdd(&(dL_drots[start_idx].z), dL_drots_start_idx.z);
        atomicAdd(&(dL_drots[start_idx].w), dL_drots_start_idx.w);

        atomicAdd(&(dL_drots[stop_idx].x), dL_drots_stop_idx.x);
        atomicAdd(&(dL_drots[stop_idx].y), dL_drots_stop_idx.y);
        atomicAdd(&(dL_drots[stop_idx].z), dL_drots_stop_idx.z);
        atomicAdd(&(dL_drots[stop_idx].w), dL_drots_stop_idx.w);

        atomicAdd(&(dL_dgeovalues[start_idx]), dL_dgeovalues_start_idx);
        atomicAdd(&(dL_dgeovalues[stop_idx]), dL_dgeovalues_stop_idx);
    }

    __global__ void __anyhit__ch() {
        unsigned long long payload_addr_high = optixGetPayload_0();
        unsigned long long payload_addr_low  = optixGetPayload_1();
        unsigned long long payload_addr = (payload_addr_high << 32) + payload_addr_low;
        OneBouncePayload* payload = reinterpret_cast<OneBouncePayload*>(payload_addr);

        auto i = optixGetPrimitiveIndex();

        if ( payload->mode == 0 ) {
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
        OneBouncePayload* payload = reinterpret_cast<OneBouncePayload*>(payload_addr);

        auto i = optixGetPrimitiveIndex();
        
        if ( i != payload->start_idx && i != payload->stop_idx) {
            // Fetch ray properties
            const float3 _rayOrigin = optixGetWorldRayOrigin();
            const float3 _rayDirection = optixGetWorldRayDirection();
            const glm::vec3 ray_org = glm::vec3(_rayOrigin.x, _rayOrigin.y, _rayOrigin.z);
            const glm::vec3 ray_dir = glm::vec3(_rayDirection.x, _rayDirection.y, _rayDirection.z);
            const auto t_min = 1E-8F;
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
                const float G = exp(- 0.5f * (u * u + v * v));
                const float geovalue = G * _geovalue;
                if ( geovalue > 0.5358F ) {
                    optixReportIntersection(t, 0, __float_as_uint(t), __float_as_uint(G), __float_as_uint(geovalue), __float_as_uint(u), __float_as_uint(v), __float_as_uint(local_ray_dir.x), __float_as_uint(local_ray_dir.y), __float_as_uint(local_ray_dir.z));
                }
            }
        }
    }
}