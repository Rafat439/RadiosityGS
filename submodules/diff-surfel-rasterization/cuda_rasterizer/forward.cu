/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, const glm::vec4* rots, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);
	dir = dir * quat_to_rotmat(rots[idx]);
	if ( dir.z < 0.0F ) dir = -dir;

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = glm::vec3(0.0F);

	float x = dir.x;
	float y = dir.y; 
	float z = dir.z;

	result += SH_C0[0] * sh[0];
	if (deg > 0)
	{
		result += SH_C1[0] * (y) * sh[1] + 
				SH_C1[1] * (z) * sh[2] + 
				SH_C1[2] * (x) * sh[3];
		if (deg > 1)
		{
			float x2 = x * x;
			float y2 = y * y;
			float z2 = z * z;
			result += SH_C2[0] * (x*y) * sh[4] + 
					SH_C2[1] * (y*z) * sh[5] + 
					SH_C2[2] * (z2 - 1.0F/3.0F) * sh[6] + 
					SH_C2[3] * (x*z) * sh[7] + 
					SH_C2[4] * (x2 - y2) * sh[8];
			if (deg > 2)
			{
				float x3 = x2 * x;
				float y3 = y2 * y;
				float z3 = z2 * z;
				result += SH_C3[0] * (x2*y - 1.0F/3.0F*y3) * sh[9] + 
						SH_C3[1] * (x*y*z) * sh[10] + 
						SH_C3[2] * (y*z2 - 1.0F/5.0F*y) * sh[11] + 
						SH_C3[3] * (z3 - 3.0F/5.0F*z) * sh[12] + 
						SH_C3[4] * (x*z2 - 1.0F/5.0F*x) * sh[13] + 
						SH_C3[5] * (z*(x - y)*(x + y)) * sh[14] + 
						SH_C3[6] * ((1.0F/3.0F)*x3 - x*y2) * sh[15];
				if (deg > 3)
				{
					float x4 = x3 * x;
					float y4 = y3 * y;
					float z4 = z3 * z;
					result += SH_C4[0] * (x3*y - x*y3) * sh[16] + 
							SH_C4[1] * (x2*y*z - 1.0F/3.0F*y3*z) * sh[17] + 
							SH_C4[2] * (x*y*z2 - 1.0F/7.0F*x*y) * sh[18] + 
							SH_C4[3] * (y*z3 - 3.0F/7.0F*y*z) * sh[19] + 
							SH_C4[4] * (z4 - 6.0F/7.0F*z2 + 3.0F/35.0F) * sh[20] + 
							SH_C4[5] * (x*z3 - 3.0F/7.0F*x*z) * sh[21] + 
							SH_C4[6] * ((1.0F/7.0F)*(x - y)*(x + y)*(7*z2 - 1)) * sh[22] + 
							SH_C4[7] * ((1.0F/3.0F)*x3*z - x*y2*z) * sh[23] + 
							SH_C4[8] * ((1.0F/6.0F)*x4 - x2*y2 + (1.0F/6.0F)*y4) * sh[24];
					if (deg > 4)
					{
						float x5 = x4 * x;
						float y5 = y4 * y;
						float z5 = z4 * z;
						result += SH_C5[0] * ((1.0F/2.0F)*x4*y - x2*y3 + (1.0F/10.0F)*y5) * sh[25] + 
								SH_C5[1] * (x*y*z*(x - y)*(x + y)) * sh[26] + 
								SH_C5[2] * ((1.0F/27.0F)*y*(3*x2 - y2)*(3*z - 1)*(3*z + 1)) * sh[27] + 
								SH_C5[3] * (x*y*z3 - 1.0F/3.0F*x*y*z) * sh[28] + 
								SH_C5[4] * ((1.0F/21.0F)*y*(21*z4 - 14*z2 + 1)) * sh[29] + 
								SH_C5[5] * ((9.0F/10.0F)*z5 - z3 + (3.0F/14.0F)*z) * sh[30] + 
								SH_C5[6] * ((1.0F/21.0F)*x*(21*z4 - 14*z2 + 1)) * sh[31] + 
								SH_C5[7] * ((1.0F/3.0F)*z*(x - y)*(x + y)*(3*z2 - 1)) * sh[32] + 
								SH_C5[8] * ((1.0F/27.0F)*x*(x2 - 3*y2)*(3*z - 1)*(3*z + 1)) * sh[33] + 
								SH_C5[9] * ((1.0F/6.0F)*x4*z - x2*y2*z + (1.0F/6.0F)*y4*z) * sh[34] + 
								SH_C5[10] * ((1.0F/10.0F)*x5 - x3*y2 + (1.0F/2.0F)*x*y4) * sh[35];
						if (deg > 5)
						{
							float x6 = x5 * x;
							float y6 = y5 * y;
							float z6 = z5 * z;
							result += SH_C6[0] * ((1.0F/10.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * sh[36] + 
									SH_C6[1] * ((1.0F/10.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[37] + 
									SH_C6[2] * ((1.0F/11.0F)*x*y*(x - y)*(x + y)*(11*z2 - 1)) * sh[38] + 
									SH_C6[3] * ((1.0F/33.0F)*y*z*(3*x2 - y2)*(11*z2 - 3)) * sh[39] + 
									SH_C6[4] * ((1.0F/33.0F)*x*y*(33*z4 - 18*z2 + 1)) * sh[40] + 
									SH_C6[5] * ((1.0F/33.0F)*y*z*(33*z4 - 30*z2 + 5)) * sh[41] + 
									SH_C6[6] * ((11.0F/15.0F)*z6 - z4 + (1.0F/3.0F)*z2 - 1.0F/63.0F) * sh[42] + 
									SH_C6[7] * ((1.0F/33.0F)*x*z*(33*z4 - 30*z2 + 5)) * sh[43] + 
									SH_C6[8] * ((1.0F/33.0F)*(x - y)*(x + y)*(33*z4 - 18*z2 + 1)) * sh[44] + 
									SH_C6[9] * ((1.0F/33.0F)*x*z*(x2 - 3*y2)*(11*z2 - 3)) * sh[45] + 
									SH_C6[10] * ((1.0F/66.0F)*(11*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[46] + 
									SH_C6[11] * ((1.0F/10.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[47] + 
									SH_C6[12] * ((1.0F/15.0F)*x6 - x4*y2 + x2*y4 - 1.0F/15.0F*y6) * sh[48];
							if (deg > 6)
							{
								float x7 = x6 * x;
								float y7 = y6 * y;
								float z7 = z6 * z;
								result += SH_C7[0] * ((1.0F/5.0F)*x6*y - x4*y3 + (3.0F/5.0F)*x2*y5 - 1.0F/35.0F*y7) * sh[49] + 
										SH_C7[1] * ((1.0F/10.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[50] + 
										SH_C7[2] * ((1.0F/130.0F)*y*(13*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[51] + 
										SH_C7[3] * ((1.0F/13.0F)*x*y*z*(x - y)*(x + y)*(13*z2 - 3)) * sh[52] + 
										SH_C7[4] * ((1.0F/429.0F)*y*(3*x2 - y2)*(143*z4 - 66*z2 + 3)) * sh[53] + 
										SH_C7[5] * ((1.0F/143.0F)*x*y*z*(143*z4 - 110*z2 + 15)) * sh[54] + 
										SH_C7[6] * ((1.0F/495.0F)*y*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[55] + 
										SH_C7[7] * ((1.0F/693.0F)*z*(429*z6 - 693*z4 + 315*z2 - 35)) * sh[56] + 
										SH_C7[8] * ((1.0F/495.0F)*x*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[57] + 
										SH_C7[9] * ((1.0F/143.0F)*z*(x - y)*(x + y)*(143*z4 - 110*z2 + 15)) * sh[58] + 
										SH_C7[10] * ((1.0F/429.0F)*x*(x2 - 3*y2)*(143*z4 - 66*z2 + 3)) * sh[59] + 
										SH_C7[11] * ((1.0F/78.0F)*z*(13*z2 - 3)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[60] + 
										SH_C7[12] * ((1.0F/130.0F)*x*(13*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[61] + 
										SH_C7[13] * ((1.0F/15.0F)*x6*z - x4*y2*z + x2*y4*z - 1.0F/15.0F*y6*z) * sh[62] + 
										SH_C7[14] * ((1.0F/35.0F)*x7 - 3.0F/5.0F*x5*y2 + x3*y4 - 1.0F/5.0F*x*y6) * sh[63];
								if (deg > 7)
								{
									float x8 = x7 * x;
									float y8 = y7 * y;
									float z8 = z7 * z;
									result += SH_C8[0] * ((1.0F/7.0F)*x7*y - x5*y3 + x3*y5 - 1.0F/7.0F*x*y7) * sh[64] + 
											SH_C8[1] * ((1.0F/35.0F)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[65] + 
											SH_C8[2] * ((1.0F/150.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)*(15*z2 - 1)) * sh[66] + 
											SH_C8[3] * ((1.0F/50.0F)*y*z*(5*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[67] + 
											SH_C8[4] * ((1.0F/65.0F)*x*y*(x - y)*(x + y)*(65*z4 - 26*z2 + 1)) * sh[68] + 
											SH_C8[5] * ((1.0F/117.0F)*y*z*(3*x2 - y2)*(39*z4 - 26*z2 + 3)) * sh[69] + 
											SH_C8[6] * ((1.0F/143.0F)*x*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[70] + 
											SH_C8[7] * ((1.0F/1001.0F)*y*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[71] + 
											SH_C8[8] * ((1.0F/12012.0F)*(6435*z8 - 12012*z6 + 6930*z4 - 1260*z2 + 35)) * sh[72] + 
											SH_C8[9] * ((1.0F/1001.0F)*x*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[73] + 
											SH_C8[10] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[74] + 
											SH_C8[11] * ((1.0F/117.0F)*x*z*(x2 - 3*y2)*(39*z4 - 26*z2 + 3)) * sh[75] + 
											SH_C8[12] * ((1.0F/390.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(65*z4 - 26*z2 + 1)) * sh[76] + 
											SH_C8[13] * ((1.0F/50.0F)*x*z*(5*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[77] + 
											SH_C8[14] * ((1.0F/225.0F)*(x - y)*(x + y)*(15*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[78] + 
											SH_C8[15] * ((1.0F/35.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[79] + 
											SH_C8[16] * ((1.0F/70.0F)*x8 - 2.0F/5.0F*x6*y2 + x4*y4 - 2.0F/5.0F*x2*y6 + (1.0F/70.0F)*y8) * sh[80];
									if (deg > 8)
									{
										float x9 = x8 * x;
										float y9 = y8 * y;
										float z9 = z8 * z;
										result += SH_C9[0] * ((1.0F/126.0F)*y*(3*x2 - y2)*(3*x6 - 27*x4*y2 + 33*x2*y4 - y6)) * sh[81] + 
												SH_C9[1] * ((1.0F/7.0F)*x7*y*z - x5*y3*z + x3*y5*z - 1.0F/7.0F*x*y7*z) * sh[82] + 
												SH_C9[2] * ((1.0F/595.0F)*y*(17*z2 - 1)*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[83] + 
												SH_C9[3] * ((1.0F/170.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 3)) * sh[84] + 
												SH_C9[4] * ((1.0F/850.0F)*y*(5*x4 - 10*x2*y2 + y4)*(85*z4 - 30*z2 + 1)) * sh[85] + 
												SH_C9[5] * ((1.0F/17.0F)*x*y*z*(x - y)*(x + y)*(17*z4 - 10*z2 + 1)) * sh[86] + 
												SH_C9[6] * ((1.0F/663.0F)*y*(3*x2 - y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[87] + 
												SH_C9[7] * ((1.0F/273.0F)*x*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[88] + 
												SH_C9[8] * ((1.0F/4004.0F)*y*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * sh[89] + 
												SH_C9[9] * ((1.0F/25740.0F)*z*(12155*z8 - 25740*z6 + 18018*z4 - 4620*z2 + 315)) * sh[90] + 
												SH_C9[10] * ((1.0F/4004.0F)*x*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * sh[91] + 
												SH_C9[11] * ((1.0F/273.0F)*z*(x - y)*(x + y)*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[92] + 
												SH_C9[12] * ((1.0F/663.0F)*x*(x2 - 3*y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[93] + 
												SH_C9[13] * ((1.0F/102.0F)*z*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(17*z4 - 10*z2 + 1)) * sh[94] + 
												SH_C9[14] * ((1.0F/850.0F)*x*(x4 - 10*x2*y2 + 5*y4)*(85*z4 - 30*z2 + 1)) * sh[95] + 
												SH_C9[15] * ((1.0F/255.0F)*z*(x - y)*(x + y)*(17*z2 - 3)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[96] + 
												SH_C9[16] * ((1.0F/595.0F)*x*(17*z2 - 1)*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[97] + 
												SH_C9[17] * ((1.0F/70.0F)*x8*z - 2.0F/5.0F*x6*y2*z + x4*y4*z - 2.0F/5.0F*x2*y6*z + (1.0F/70.0F)*y8*z) * sh[98] + 
												SH_C9[18] * ((1.0F/126.0F)*x*(x2 - 3*y2)*(x6 - 33*x4*y2 + 27*x2*y4 - 3*y6)) * sh[99];
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (fabs(d) < 1E-8F) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* geovalues,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float3* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_geovalue,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}
	float cos = -sumf3(p_view * normal);
	if (fabs(cos) < 1E-8F) return;
	float multiplier = cos > 0 ? 1: -1;

#if DUAL_VISIBLE
	normal = multiplier * normal;
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the geovalue of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(geovalues[idx]), 0.000001));
#else
	float cutoff = 3.0f;
#endif

	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(extent.x, extent.y));
		// radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, rotations, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = {point_image.x, point_image.y, multiplier};
	normal_geovalue[idx] = {normal.x, normal.y, normal.z, geovalues[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float3* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_geovalue,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others, 
	int* __restrict__ count_accum, 
	float* __restrict__ weight_accum)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_geovalue[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float back_T = 1.0f;
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_geovalue[block.thread_rank()] = normal_geovalue[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float3 xym = collected_xy[j];
			const float2 xy = { xym.x, xym.y };
			const bool visible = xym.z > 0;
			// if (!visible) continue;
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			// Transform the two planes into local u-v system. 
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			// Cross product of two planes is a line, Eq. (9)
			float3 p = cross(k, l);
			if (fabs(p.z) < 1E-8F) continue;
			// Perspective division to get the intersection (u,v), Eq. (10)
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			// Add low pass filter
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d);

			// compute depth
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			// if a point is too small, its depth is not reliable?
			depth = (rho3d <= rho2d) ? depth : Tw.z;
			if (depth < near_n) continue;

			float4 nor_o = collected_normal_geovalue[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float geovalue = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian geovalue
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, footprint_activation(geovalue * exp(power)));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;


			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;

			atomicAdd(&count_accum[collected_id[j]], 1);
			atomicAdd(&weight_accum[collected_id[j]], w);
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			if ( visible ) {
				for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			} else back_T *= (1 - alpha);
			
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + BACK_FACE_OFFSET * H * W] = 1 - back_T;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float3* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_geovalue,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others, 
	int* count_accum, 
	float* weight_accum)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_geovalue,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others, 
		count_accum, 
		weight_accum);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* geovalues,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float3* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_geovalue,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		geovalues,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_geovalue,
		grid,
		tiles_touched,
		prefiltered
		);
}
