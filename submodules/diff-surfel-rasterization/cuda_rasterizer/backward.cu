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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, const glm::vec4* rots, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec4* dL_drots, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 raw_dir = dir_orig / glm::length(dir_orig);
	glm::mat3 rot = quat_to_rotmat(rots[idx]);
	glm::vec3 dir = raw_dir * rot;
	float dir_multiplier = 1.0F;
	if ( dir.z < 0.0F ) dir_multiplier = -1.0F;
	dir = dir_multiplier * dir;

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	dL_dsh[0] = SH_C0[0] * dL_dRGB;
	
	if (deg > 0)
	{
		dL_dsh[1] = SH_C1[0] * (y) * dL_dRGB;
		dL_dsh[2] = SH_C1[1] * (z) * dL_dRGB;
		dL_dsh[3] = SH_C1[2] * (x) * dL_dRGB;

		dRGBdx += SH_C1[2] * sh[3];
		dRGBdy += SH_C1[0] * sh[1];
		dRGBdz += SH_C1[1] * sh[2];

		if (deg > 1)
		{
			float x2 = x * x;
			float y2 = y * y;
			float z2 = z * z;
			dL_dsh[4] = SH_C2[0] * (x*y) * dL_dRGB;
			dL_dsh[5] = SH_C2[1] * (y*z) * dL_dRGB;
			dL_dsh[6] = SH_C2[2] * (z2 - 1.0F/3.0F) * dL_dRGB;
			dL_dsh[7] = SH_C2[3] * (x*z) * dL_dRGB;
			dL_dsh[8] = SH_C2[4] * (x2 - y2) * dL_dRGB;

			dRGBdx += SH_C2[0] * (y) * sh[4] + SH_C2[3] * (z) * sh[7] + SH_C2[4] * (2*x) * sh[8];
			dRGBdy += SH_C2[0] * (x) * sh[4] + SH_C2[1] * (z) * sh[5] + SH_C2[4] * (-2*y) * sh[8];
			dRGBdz += SH_C2[1] * (y) * sh[5] + SH_C2[2] * (2*z) * sh[6] + SH_C2[3] * (x) * sh[7];

			if (deg > 2)
			{
				float x3 = x2 * x;
				float y3 = y2 * y;
				float z3 = z2 * z;
				dL_dsh[9] = SH_C3[0] * (x2*y - 1.0F/3.0F*y3) * dL_dRGB;
				dL_dsh[10] = SH_C3[1] * (x*y*z) * dL_dRGB;
				dL_dsh[11] = SH_C3[2] * (y*z2 - 1.0F/5.0F*y) * dL_dRGB;
				dL_dsh[12] = SH_C3[3] * (z3 - 3.0F/5.0F*z) * dL_dRGB;
				dL_dsh[13] = SH_C3[4] * (x*z2 - 1.0F/5.0F*x) * dL_dRGB;
				dL_dsh[14] = SH_C3[5] * (z*(x - y)*(x + y)) * dL_dRGB;
				dL_dsh[15] = SH_C3[6] * ((1.0F/3.0F)*x3 - x*y2) * dL_dRGB;

				dRGBdx += SH_C3[0] * (2*x*y) * sh[9] + SH_C3[1] * (y*z) * sh[10] + SH_C3[4] * (z2 - 1.0F/5.0F) * sh[13] + SH_C3[5] * (2*x*z) * sh[14] + SH_C3[6] * (x2 - y2) * sh[15];
				dRGBdy += SH_C3[0] * (x2 - y2) * sh[9] + SH_C3[1] * (x*z) * sh[10] + SH_C3[2] * (z2 - 1.0F/5.0F) * sh[11] + SH_C3[5] * (-2*y*z) * sh[14] + SH_C3[6] * (-2*x*y) * sh[15];
				dRGBdz += SH_C3[1] * (x*y) * sh[10] + SH_C3[2] * (2*y*z) * sh[11] + SH_C3[3] * (3*z2 - 3.0F/5.0F) * sh[12] + SH_C3[4] * (2*x*z) * sh[13] + SH_C3[5] * (x2 - y2) * sh[14];

				if (deg > 3)
				{
					float x4 = x3 * x;
					float y4 = y3 * y;
					float z4 = z3 * z;
					dL_dsh[16] = SH_C4[0] * (x3*y - x*y3) * dL_dRGB;
					dL_dsh[17] = SH_C4[1] * (x2*y*z - 1.0F/3.0F*y3*z) * dL_dRGB;
					dL_dsh[18] = SH_C4[2] * (x*y*z2 - 1.0F/7.0F*x*y) * dL_dRGB;
					dL_dsh[19] = SH_C4[3] * (y*z3 - 3.0F/7.0F*y*z) * dL_dRGB;
					dL_dsh[20] = SH_C4[4] * (z4 - 6.0F/7.0F*z2 + 3.0F/35.0F) * dL_dRGB;
					dL_dsh[21] = SH_C4[5] * (x*z3 - 3.0F/7.0F*x*z) * dL_dRGB;
					dL_dsh[22] = SH_C4[6] * ((1.0F/7.0F)*(x - y)*(x + y)*(7*z2 - 1)) * dL_dRGB;
					dL_dsh[23] = SH_C4[7] * ((1.0F/3.0F)*x3*z - x*y2*z) * dL_dRGB;
					dL_dsh[24] = SH_C4[8] * ((1.0F/6.0F)*x4 - x2*y2 + (1.0F/6.0F)*y4) * dL_dRGB;

					dRGBdx += SH_C4[0] * (3*x2*y - y3) * sh[16] + SH_C4[1] * (2*x*y*z) * sh[17] + SH_C4[2] * (y*z2 - 1.0F/7.0F*y) * sh[18] + SH_C4[5] * (z3 - 3.0F/7.0F*z) * sh[21] + SH_C4[6] * (2*x*z2 - 2.0F/7.0F*x) * sh[22] + SH_C4[7] * (z*(x - y)*(x + y)) * sh[23] + SH_C4[8] * ((2.0F/3.0F)*x3 - 2*x*y2) * sh[24];
					dRGBdy += SH_C4[0] * (x3 - 3*x*y2) * sh[16] + SH_C4[1] * (z*(x - y)*(x + y)) * sh[17] + SH_C4[2] * (x*z2 - 1.0F/7.0F*x) * sh[18] + SH_C4[3] * (z3 - 3.0F/7.0F*z) * sh[19] + SH_C4[6] * (-2*y*z2 + (2.0F/7.0F)*y) * sh[22] + SH_C4[7] * (-2*x*y*z) * sh[23] + SH_C4[8] * (-2*x2*y + (2.0F/3.0F)*y3) * sh[24];
					dRGBdz += SH_C4[1] * (x2*y - 1.0F/3.0F*y3) * sh[17] + SH_C4[2] * (2*x*y*z) * sh[18] + SH_C4[3] * (3*y*z2 - 3.0F/7.0F*y) * sh[19] + SH_C4[4] * (4*z3 - 12.0F/7.0F*z) * sh[20] + SH_C4[5] * (3*x*z2 - 3.0F/7.0F*x) * sh[21] + SH_C4[6] * (2*z*(x - y)*(x + y)) * sh[22] + SH_C4[7] * ((1.0F/3.0F)*x3 - x*y2) * sh[23];

					if (deg > 4)
					{
						float x5 = x4 * x;
						float y5 = y4 * y;
						float z5 = z4 * z;
						dL_dsh[25] = SH_C5[0] * ((1.0F/2.0F)*x4*y - x2*y3 + (1.0F/10.0F)*y5) * dL_dRGB;
						dL_dsh[26] = SH_C5[1] * (x*y*z*(x - y)*(x + y)) * dL_dRGB;
						dL_dsh[27] = SH_C5[2] * ((1.0F/27.0F)*y*(3*x2 - y2)*(3*z - 1)*(3*z + 1)) * dL_dRGB;
						dL_dsh[28] = SH_C5[3] * (x*y*z3 - 1.0F/3.0F*x*y*z) * dL_dRGB;
						dL_dsh[29] = SH_C5[4] * ((1.0F/21.0F)*y*(21*z4 - 14*z2 + 1)) * dL_dRGB;
						dL_dsh[30] = SH_C5[5] * ((9.0F/10.0F)*z5 - z3 + (3.0F/14.0F)*z) * dL_dRGB;
						dL_dsh[31] = SH_C5[6] * ((1.0F/21.0F)*x*(21*z4 - 14*z2 + 1)) * dL_dRGB;
						dL_dsh[32] = SH_C5[7] * ((1.0F/3.0F)*z*(x - y)*(x + y)*(3*z2 - 1)) * dL_dRGB;
						dL_dsh[33] = SH_C5[8] * ((1.0F/27.0F)*x*(x2 - 3*y2)*(3*z - 1)*(3*z + 1)) * dL_dRGB;
						dL_dsh[34] = SH_C5[9] * ((1.0F/6.0F)*x4*z - x2*y2*z + (1.0F/6.0F)*y4*z) * dL_dRGB;
						dL_dsh[35] = SH_C5[10] * ((1.0F/10.0F)*x5 - x3*y2 + (1.0F/2.0F)*x*y4) * dL_dRGB;

						dRGBdx += SH_C5[0] * (2*x*y*(x - y)*(x + y)) * sh[25] + SH_C5[1] * (y*z*(3*x2 - y2)) * sh[26] + SH_C5[2] * (2*x*y*z2 - 2.0F/9.0F*x*y) * sh[27] + SH_C5[3] * (y*z3 - 1.0F/3.0F*y*z) * sh[28] + SH_C5[6] * (z4 - 2.0F/3.0F*z2 + 1.0F/21.0F) * sh[31] + SH_C5[7] * (2*x*z3 - 2.0F/3.0F*x*z) * sh[32] + SH_C5[8] * ((1.0F/9.0F)*(x - y)*(x + y)*(3*z - 1)*(3*z + 1)) * sh[33] + SH_C5[9] * ((2.0F/3.0F)*x*z*(x2 - 3*y2)) * sh[34] + SH_C5[10] * ((1.0F/2.0F)*x4 - 3*x2*y2 + (1.0F/2.0F)*y4) * sh[35];
						dRGBdy += SH_C5[0] * ((1.0F/2.0F)*x4 - 3*x2*y2 + (1.0F/2.0F)*y4) * sh[25] + SH_C5[1] * (x*z*(x2 - 3*y2)) * sh[26] + SH_C5[2] * ((1.0F/9.0F)*(x - y)*(x + y)*(3*z - 1)*(3*z + 1)) * sh[27] + SH_C5[3] * (x*z3 - 1.0F/3.0F*x*z) * sh[28] + SH_C5[4] * (z4 - 2.0F/3.0F*z2 + 1.0F/21.0F) * sh[29] + SH_C5[7] * (-2*y*z3 + (2.0F/3.0F)*y*z) * sh[32] + SH_C5[8] * (-2*x*y*z2 + (2.0F/9.0F)*x*y) * sh[33] + SH_C5[9] * (-2*x2*y*z + (2.0F/3.0F)*y3*z) * sh[34] + SH_C5[10] * (-2*x3*y + 2*x*y3) * sh[35];
						dRGBdz += SH_C5[1] * (x3*y - x*y3) * sh[26] + SH_C5[2] * ((2.0F/3.0F)*y*z*(3*x2 - y2)) * sh[27] + SH_C5[3] * (3*x*y*z2 - 1.0F/3.0F*x*y) * sh[28] + SH_C5[4] * (4*y*z3 - 4.0F/3.0F*y*z) * sh[29] + SH_C5[5] * ((3.0F/14.0F)*(21*z4 - 14*z2 + 1)) * sh[30] + SH_C5[6] * (4*x*z3 - 4.0F/3.0F*x*z) * sh[31] + SH_C5[7] * ((1.0F/3.0F)*(x - y)*(x + y)*(3*z - 1)*(3*z + 1)) * sh[32] + SH_C5[8] * ((2.0F/3.0F)*x*z*(x2 - 3*y2)) * sh[33] + SH_C5[9] * ((1.0F/6.0F)*x4 - x2*y2 + (1.0F/6.0F)*y4) * sh[34];

						if (deg > 5)
						{
							float x6 = x5 * x;
							float y6 = y5 * y;
							float z6 = z5 * z;
							dL_dsh[36] = SH_C6[0] * ((1.0F/10.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * dL_dRGB;
							dL_dsh[37] = SH_C6[1] * ((1.0F/10.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * dL_dRGB;
							dL_dsh[38] = SH_C6[2] * ((1.0F/11.0F)*x*y*(x - y)*(x + y)*(11*z2 - 1)) * dL_dRGB;
							dL_dsh[39] = SH_C6[3] * ((1.0F/33.0F)*y*z*(3*x2 - y2)*(11*z2 - 3)) * dL_dRGB;
							dL_dsh[40] = SH_C6[4] * ((1.0F/33.0F)*x*y*(33*z4 - 18*z2 + 1)) * dL_dRGB;
							dL_dsh[41] = SH_C6[5] * ((1.0F/33.0F)*y*z*(33*z4 - 30*z2 + 5)) * dL_dRGB;
							dL_dsh[42] = SH_C6[6] * ((11.0F/15.0F)*z6 - z4 + (1.0F/3.0F)*z2 - 1.0F/63.0F) * dL_dRGB;
							dL_dsh[43] = SH_C6[7] * ((1.0F/33.0F)*x*z*(33*z4 - 30*z2 + 5)) * dL_dRGB;
							dL_dsh[44] = SH_C6[8] * ((1.0F/33.0F)*(x - y)*(x + y)*(33*z4 - 18*z2 + 1)) * dL_dRGB;
							dL_dsh[45] = SH_C6[9] * ((1.0F/33.0F)*x*z*(x2 - 3*y2)*(11*z2 - 3)) * dL_dRGB;
							dL_dsh[46] = SH_C6[10] * ((1.0F/66.0F)*(11*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * dL_dRGB;
							dL_dsh[47] = SH_C6[11] * ((1.0F/10.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * dL_dRGB;
							dL_dsh[48] = SH_C6[12] * ((1.0F/15.0F)*x6 - x4*y2 + x2*y4 - 1.0F/15.0F*y6) * dL_dRGB;

							dRGBdx += SH_C6[0] * ((3.0F/10.0F)*y*(5*x4 - 10*x2*y2 + y4)) * sh[36] + SH_C6[1] * (2*x*y*z*(x - y)*(x + y)) * sh[37] + SH_C6[2] * ((1.0F/11.0F)*y*(3*x2 - y2)*(11*z2 - 1)) * sh[38] + SH_C6[3] * ((2.0F/11.0F)*x*y*z*(11*z2 - 3)) * sh[39] + SH_C6[4] * ((1.0F/33.0F)*y*(33*z4 - 18*z2 + 1)) * sh[40] + SH_C6[7] * (z5 - 10.0F/11.0F*z3 + (5.0F/33.0F)*z) * sh[43] + SH_C6[8] * ((2.0F/33.0F)*x*(33*z4 - 18*z2 + 1)) * sh[44] + SH_C6[9] * ((1.0F/11.0F)*z*(x - y)*(x + y)*(11*z2 - 3)) * sh[45] + SH_C6[10] * ((2.0F/33.0F)*x*(x2 - 3*y2)*(11*z2 - 1)) * sh[46] + SH_C6[11] * ((1.0F/2.0F)*x4*z - 3*x2*y2*z + (1.0F/2.0F)*y4*z) * sh[47] + SH_C6[12] * ((2.0F/5.0F)*x5 - 4*x3*y2 + 2*x*y4) * sh[48];
							dRGBdy += SH_C6[0] * ((3.0F/10.0F)*x*(x4 - 10*x2*y2 + 5*y4)) * sh[36] + SH_C6[1] * ((1.0F/2.0F)*x4*z - 3*x2*y2*z + (1.0F/2.0F)*y4*z) * sh[37] + SH_C6[2] * ((1.0F/11.0F)*x*(x2 - 3*y2)*(11*z2 - 1)) * sh[38] + SH_C6[3] * ((1.0F/11.0F)*z*(x - y)*(x + y)*(11*z2 - 3)) * sh[39] + SH_C6[4] * ((1.0F/33.0F)*x*(33*z4 - 18*z2 + 1)) * sh[40] + SH_C6[5] * (z5 - 10.0F/11.0F*z3 + (5.0F/33.0F)*z) * sh[41] + SH_C6[8] * (-2.0F/33.0F*y*(33*z4 - 18*z2 + 1)) * sh[44] + SH_C6[9] * (-2*x*y*z3 + (6.0F/11.0F)*x*y*z) * sh[45] + SH_C6[10] * (-2.0F/33.0F*y*(3*x2 - y2)*(11*z2 - 1)) * sh[46] + SH_C6[11] * (-2*x*y*z*(x - y)*(x + y)) * sh[47] + SH_C6[12] * (-2*x4*y + 4*x2*y3 - 2.0F/5.0F*y5) * sh[48];
							dRGBdz += SH_C6[1] * ((1.0F/2.0F)*x4*y - x2*y3 + (1.0F/10.0F)*y5) * sh[37] + SH_C6[2] * (2*x*y*z*(x - y)*(x + y)) * sh[38] + SH_C6[3] * ((1.0F/11.0F)*y*(3*x2 - y2)*(11*z2 - 1)) * sh[39] + SH_C6[4] * ((4.0F/11.0F)*x*y*z*(11*z2 - 3)) * sh[40] + SH_C6[5] * ((5.0F/33.0F)*y*(33*z4 - 18*z2 + 1)) * sh[41] + SH_C6[6] * ((22.0F/5.0F)*z5 - 4*z3 + (2.0F/3.0F)*z) * sh[42] + SH_C6[7] * ((5.0F/33.0F)*x*(33*z4 - 18*z2 + 1)) * sh[43] + SH_C6[8] * ((4.0F/11.0F)*z*(x - y)*(x + y)*(11*z2 - 3)) * sh[44] + SH_C6[9] * ((1.0F/11.0F)*x*(x2 - 3*y2)*(11*z2 - 1)) * sh[45] + SH_C6[10] * ((1.0F/3.0F)*x4*z - 2*x2*y2*z + (1.0F/3.0F)*y4*z) * sh[46] + SH_C6[11] * ((1.0F/10.0F)*x5 - x3*y2 + (1.0F/2.0F)*x*y4) * sh[47];

							if (deg > 6)
							{
								float x7 = x6 * x;
								float y7 = y6 * y;
								float z7 = z6 * z;
								dL_dsh[49] = SH_C7[0] * ((1.0F/5.0F)*x6*y - x4*y3 + (3.0F/5.0F)*x2*y5 - 1.0F/35.0F*y7) * dL_dRGB;
								dL_dsh[50] = SH_C7[1] * ((1.0F/10.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * dL_dRGB;
								dL_dsh[51] = SH_C7[2] * ((1.0F/130.0F)*y*(13*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * dL_dRGB;
								dL_dsh[52] = SH_C7[3] * ((1.0F/13.0F)*x*y*z*(x - y)*(x + y)*(13*z2 - 3)) * dL_dRGB;
								dL_dsh[53] = SH_C7[4] * ((1.0F/429.0F)*y*(3*x2 - y2)*(143*z4 - 66*z2 + 3)) * dL_dRGB;
								dL_dsh[54] = SH_C7[5] * ((1.0F/143.0F)*x*y*z*(143*z4 - 110*z2 + 15)) * dL_dRGB;
								dL_dsh[55] = SH_C7[6] * ((1.0F/495.0F)*y*(429*z6 - 495*z4 + 135*z2 - 5)) * dL_dRGB;
								dL_dsh[56] = SH_C7[7] * ((1.0F/693.0F)*z*(429*z6 - 693*z4 + 315*z2 - 35)) * dL_dRGB;
								dL_dsh[57] = SH_C7[8] * ((1.0F/495.0F)*x*(429*z6 - 495*z4 + 135*z2 - 5)) * dL_dRGB;
								dL_dsh[58] = SH_C7[9] * ((1.0F/143.0F)*z*(x - y)*(x + y)*(143*z4 - 110*z2 + 15)) * dL_dRGB;
								dL_dsh[59] = SH_C7[10] * ((1.0F/429.0F)*x*(x2 - 3*y2)*(143*z4 - 66*z2 + 3)) * dL_dRGB;
								dL_dsh[60] = SH_C7[11] * ((1.0F/78.0F)*z*(13*z2 - 3)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * dL_dRGB;
								dL_dsh[61] = SH_C7[12] * ((1.0F/130.0F)*x*(13*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * dL_dRGB;
								dL_dsh[62] = SH_C7[13] * ((1.0F/15.0F)*x6*z - x4*y2*z + x2*y4*z - 1.0F/15.0F*y6*z) * dL_dRGB;
								dL_dsh[63] = SH_C7[14] * ((1.0F/35.0F)*x7 - 3.0F/5.0F*x5*y2 + x3*y4 - 1.0F/5.0F*x*y6) * dL_dRGB;

								dRGBdx += SH_C7[0] * ((2.0F/5.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * sh[49] + SH_C7[1] * ((3.0F/10.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[50] + SH_C7[2] * ((2.0F/13.0F)*x*y*(x - y)*(x + y)*(13*z2 - 1)) * sh[51] + SH_C7[3] * ((1.0F/13.0F)*y*z*(3*x2 - y2)*(13*z2 - 3)) * sh[52] + SH_C7[4] * ((2.0F/143.0F)*x*y*(143*z4 - 66*z2 + 3)) * sh[53] + SH_C7[5] * ((1.0F/143.0F)*y*z*(143*z4 - 110*z2 + 15)) * sh[54] + SH_C7[8] * ((1.0F/495.0F)*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[57] + SH_C7[9] * ((2.0F/143.0F)*x*z*(143*z4 - 110*z2 + 15)) * sh[58] + SH_C7[10] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z4 - 66*z2 + 3)) * sh[59] + SH_C7[11] * ((2.0F/39.0F)*x*z*(x2 - 3*y2)*(13*z2 - 3)) * sh[60] + SH_C7[12] * ((1.0F/26.0F)*(13*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[61] + SH_C7[13] * ((2.0F/5.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[62] + SH_C7[14] * ((1.0F/5.0F)*x6 - 3*x4*y2 + 3*x2*y4 - 1.0F/5.0F*y6) * sh[63];
								dRGBdy += SH_C7[0] * ((1.0F/5.0F)*x6 - 3*x4*y2 + 3*x2*y4 - 1.0F/5.0F*y6) * sh[49] + SH_C7[1] * ((3.0F/10.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[50] + SH_C7[2] * ((1.0F/26.0F)*(13*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[51] + SH_C7[3] * ((1.0F/13.0F)*x*z*(x2 - 3*y2)*(13*z2 - 3)) * sh[52] + SH_C7[4] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z4 - 66*z2 + 3)) * sh[53] + SH_C7[5] * ((1.0F/143.0F)*x*z*(143*z4 - 110*z2 + 15)) * sh[54] + SH_C7[6] * ((1.0F/495.0F)*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[55] + SH_C7[9] * (-2.0F/143.0F*y*z*(143*z4 - 110*z2 + 15)) * sh[58] + SH_C7[10] * (-2.0F/143.0F*x*y*(143*z4 - 66*z2 + 3)) * sh[59] + SH_C7[11] * (-2.0F/39.0F*y*z*(3*x2 - y2)*(13*z2 - 3)) * sh[60] + SH_C7[12] * (-2.0F/13.0F*x*y*(x - y)*(x + y)*(13*z2 - 1)) * sh[61] + SH_C7[13] * (-2.0F/5.0F*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[62] + SH_C7[14] * (-6.0F/5.0F*x5*y + 4*x3*y3 - 6.0F/5.0F*x*y5) * sh[63];
								dRGBdz += SH_C7[1] * ((1.0F/10.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * sh[50] + SH_C7[2] * ((1.0F/5.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[51] + SH_C7[3] * ((3.0F/13.0F)*x*y*(x - y)*(x + y)*(13*z2 - 1)) * sh[52] + SH_C7[4] * ((4.0F/39.0F)*y*z*(3*x2 - y2)*(13*z2 - 3)) * sh[53] + SH_C7[5] * ((5.0F/143.0F)*x*y*(143*z4 - 66*z2 + 3)) * sh[54] + SH_C7[6] * ((2.0F/55.0F)*y*z*(143*z4 - 110*z2 + 15)) * sh[55] + SH_C7[7] * ((1.0F/99.0F)*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[56] + SH_C7[8] * ((2.0F/55.0F)*x*z*(143*z4 - 110*z2 + 15)) * sh[57] + SH_C7[9] * ((5.0F/143.0F)*(x - y)*(x + y)*(143*z4 - 66*z2 + 3)) * sh[58] + SH_C7[10] * ((4.0F/39.0F)*x*z*(x2 - 3*y2)*(13*z2 - 3)) * sh[59] + SH_C7[11] * ((1.0F/26.0F)*(13*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[60] + SH_C7[12] * ((1.0F/5.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[61] + SH_C7[13] * ((1.0F/15.0F)*x6 - x4*y2 + x2*y4 - 1.0F/15.0F*y6) * sh[62];

								if (deg > 7)
								{
									float x8 = x7 * x;
									float y8 = y7 * y;
									float z8 = z7 * z;
									dL_dsh[64] = SH_C8[0] * ((1.0F/7.0F)*x7*y - x5*y3 + x3*y5 - 1.0F/7.0F*x*y7) * dL_dRGB;
									dL_dsh[65] = SH_C8[1] * ((1.0F/35.0F)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * dL_dRGB;
									dL_dsh[66] = SH_C8[2] * ((1.0F/150.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)*(15*z2 - 1)) * dL_dRGB;
									dL_dsh[67] = SH_C8[3] * ((1.0F/50.0F)*y*z*(5*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * dL_dRGB;
									dL_dsh[68] = SH_C8[4] * ((1.0F/65.0F)*x*y*(x - y)*(x + y)*(65*z4 - 26*z2 + 1)) * dL_dRGB;
									dL_dsh[69] = SH_C8[5] * ((1.0F/117.0F)*y*z*(3*x2 - y2)*(39*z4 - 26*z2 + 3)) * dL_dRGB;
									dL_dsh[70] = SH_C8[6] * ((1.0F/143.0F)*x*y*(143*z6 - 143*z4 + 33*z2 - 1)) * dL_dRGB;
									dL_dsh[71] = SH_C8[7] * ((1.0F/1001.0F)*y*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * dL_dRGB;
									dL_dsh[72] = SH_C8[8] * ((1.0F/12012.0F)*(6435*z8 - 12012*z6 + 6930*z4 - 1260*z2 + 35)) * dL_dRGB;
									dL_dsh[73] = SH_C8[9] * ((1.0F/1001.0F)*x*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * dL_dRGB;
									dL_dsh[74] = SH_C8[10] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z6 - 143*z4 + 33*z2 - 1)) * dL_dRGB;
									dL_dsh[75] = SH_C8[11] * ((1.0F/117.0F)*x*z*(x2 - 3*y2)*(39*z4 - 26*z2 + 3)) * dL_dRGB;
									dL_dsh[76] = SH_C8[12] * ((1.0F/390.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(65*z4 - 26*z2 + 1)) * dL_dRGB;
									dL_dsh[77] = SH_C8[13] * ((1.0F/50.0F)*x*z*(5*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * dL_dRGB;
									dL_dsh[78] = SH_C8[14] * ((1.0F/225.0F)*(x - y)*(x + y)*(15*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * dL_dRGB;
									dL_dsh[79] = SH_C8[15] * ((1.0F/35.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * dL_dRGB;
									dL_dsh[80] = SH_C8[16] * ((1.0F/70.0F)*x8 - 2.0F/5.0F*x6*y2 + x4*y4 - 2.0F/5.0F*x2*y6 + (1.0F/70.0F)*y8) * dL_dRGB;

									dRGBdx += SH_C8[0] * (x6*y - 5*x4*y3 + 3*x2*y5 - 1.0F/7.0F*y7) * sh[64] + SH_C8[1] * ((2.0F/5.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[65] + SH_C8[2] * ((1.0F/50.0F)*y*(15*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[66] + SH_C8[3] * ((2.0F/5.0F)*x*y*z*(x - y)*(x + y)*(5*z2 - 1)) * sh[67] + SH_C8[4] * ((1.0F/65.0F)*y*(3*x2 - y2)*(65*z4 - 26*z2 + 1)) * sh[68] + SH_C8[5] * ((2.0F/39.0F)*x*y*z*(39*z4 - 26*z2 + 3)) * sh[69] + SH_C8[6] * ((1.0F/143.0F)*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[70] + SH_C8[9] * ((1.0F/1001.0F)*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[73] + SH_C8[10] * ((2.0F/143.0F)*x*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[74] + SH_C8[11] * ((1.0F/39.0F)*z*(x - y)*(x + y)*(39*z4 - 26*z2 + 3)) * sh[75] + SH_C8[12] * ((2.0F/195.0F)*x*(x2 - 3*y2)*(65*z4 - 26*z2 + 1)) * sh[76] + SH_C8[13] * ((1.0F/10.0F)*z*(5*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[77] + SH_C8[14] * ((2.0F/75.0F)*x*(15*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[78] + SH_C8[15] * ((1.0F/5.0F)*x6*z - 3*x4*y2*z + 3*x2*y4*z - 1.0F/5.0F*y6*z) * sh[79] + SH_C8[16] * ((4.0F/35.0F)*x*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[80];
									dRGBdy += SH_C8[0] * ((1.0F/7.0F)*x7 - 3*x5*y2 + 5*x3*y4 - x*y6) * sh[64] + SH_C8[1] * ((1.0F/5.0F)*x6*z - 3*x4*y2*z + 3*x2*y4*z - 1.0F/5.0F*y6*z) * sh[65] + SH_C8[2] * ((1.0F/50.0F)*x*(15*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[66] + SH_C8[3] * ((1.0F/10.0F)*z*(5*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[67] + SH_C8[4] * ((1.0F/65.0F)*x*(x2 - 3*y2)*(65*z4 - 26*z2 + 1)) * sh[68] + SH_C8[5] * ((1.0F/39.0F)*z*(x - y)*(x + y)*(39*z4 - 26*z2 + 3)) * sh[69] + SH_C8[6] * ((1.0F/143.0F)*x*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[70] + SH_C8[7] * ((1.0F/1001.0F)*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[71] + SH_C8[10] * (-2.0F/143.0F*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[74] + SH_C8[11] * (-2.0F/39.0F*x*y*z*(39*z4 - 26*z2 + 3)) * sh[75] + SH_C8[12] * (-2.0F/195.0F*y*(3*x2 - y2)*(65*z4 - 26*z2 + 1)) * sh[76] + SH_C8[13] * (-2.0F/5.0F*x*y*z*(x - y)*(x + y)*(5*z2 - 1)) * sh[77] + SH_C8[14] * (-2.0F/75.0F*y*(15*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[78] + SH_C8[15] * (-2.0F/5.0F*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[79] + SH_C8[16] * (-4.0F/35.0F*y*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[80];
									dRGBdz += SH_C8[1] * ((1.0F/5.0F)*x6*y - x4*y3 + (3.0F/5.0F)*x2*y5 - 1.0F/35.0F*y7) * sh[65] + SH_C8[2] * ((1.0F/5.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[66] + SH_C8[3] * ((1.0F/50.0F)*y*(15*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[67] + SH_C8[4] * ((4.0F/5.0F)*x*y*z*(x - y)*(x + y)*(5*z2 - 1)) * sh[68] + SH_C8[5] * ((1.0F/39.0F)*y*(3*x2 - y2)*(65*z4 - 26*z2 + 1)) * sh[69] + SH_C8[6] * ((2.0F/13.0F)*x*y*z*(39*z4 - 26*z2 + 3)) * sh[70] + SH_C8[7] * ((5.0F/143.0F)*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[71] + SH_C8[8] * ((6.0F/1001.0F)*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[72] + SH_C8[9] * ((5.0F/143.0F)*x*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[73] + SH_C8[10] * ((2.0F/13.0F)*z*(x - y)*(x + y)*(39*z4 - 26*z2 + 3)) * sh[74] + SH_C8[11] * ((1.0F/39.0F)*x*(x2 - 3*y2)*(65*z4 - 26*z2 + 1)) * sh[75] + SH_C8[12] * ((2.0F/15.0F)*z*(5*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[76] + SH_C8[13] * ((1.0F/50.0F)*x*(15*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[77] + SH_C8[14] * ((2.0F/15.0F)*z*(x - y)*(x + y)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[78] + SH_C8[15] * ((1.0F/35.0F)*x7 - 3.0F/5.0F*x5*y2 + x3*y4 - 1.0F/5.0F*x*y6) * sh[79];

									if (deg > 8)
									{
										float x9 = x8 * x;
										float y9 = y8 * y;
										float z9 = z8 * z;
										dL_dsh[81] = SH_C9[0] * ((1.0F/126.0F)*y*(3*x2 - y2)*(3*x6 - 27*x4*y2 + 33*x2*y4 - y6)) * dL_dRGB;
										dL_dsh[82] = SH_C9[1] * ((1.0F/7.0F)*x7*y*z - x5*y3*z + x3*y5*z - 1.0F/7.0F*x*y7*z) * dL_dRGB;
										dL_dsh[83] = SH_C9[2] * ((1.0F/595.0F)*y*(17*z2 - 1)*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * dL_dRGB;
										dL_dsh[84] = SH_C9[3] * ((1.0F/170.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 3)) * dL_dRGB;
										dL_dsh[85] = SH_C9[4] * ((1.0F/850.0F)*y*(5*x4 - 10*x2*y2 + y4)*(85*z4 - 30*z2 + 1)) * dL_dRGB;
										dL_dsh[86] = SH_C9[5] * ((1.0F/17.0F)*x*y*z*(x - y)*(x + y)*(17*z4 - 10*z2 + 1)) * dL_dRGB;
										dL_dsh[87] = SH_C9[6] * ((1.0F/663.0F)*y*(3*x2 - y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * dL_dRGB;
										dL_dsh[88] = SH_C9[7] * ((1.0F/273.0F)*x*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * dL_dRGB;
										dL_dsh[89] = SH_C9[8] * ((1.0F/4004.0F)*y*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * dL_dRGB;
										dL_dsh[90] = SH_C9[9] * ((1.0F/25740.0F)*z*(12155*z8 - 25740*z6 + 18018*z4 - 4620*z2 + 315)) * dL_dRGB;
										dL_dsh[91] = SH_C9[10] * ((1.0F/4004.0F)*x*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * dL_dRGB;
										dL_dsh[92] = SH_C9[11] * ((1.0F/273.0F)*z*(x - y)*(x + y)*(221*z6 - 273*z4 + 91*z2 - 7)) * dL_dRGB;
										dL_dsh[93] = SH_C9[12] * ((1.0F/663.0F)*x*(x2 - 3*y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * dL_dRGB;
										dL_dsh[94] = SH_C9[13] * ((1.0F/102.0F)*z*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(17*z4 - 10*z2 + 1)) * dL_dRGB;
										dL_dsh[95] = SH_C9[14] * ((1.0F/850.0F)*x*(x4 - 10*x2*y2 + 5*y4)*(85*z4 - 30*z2 + 1)) * dL_dRGB;
										dL_dsh[96] = SH_C9[15] * ((1.0F/255.0F)*z*(x - y)*(x + y)*(17*z2 - 3)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * dL_dRGB;
										dL_dsh[97] = SH_C9[16] * ((1.0F/595.0F)*x*(17*z2 - 1)*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * dL_dRGB;
										dL_dsh[98] = SH_C9[17] * ((1.0F/70.0F)*x8*z - 2.0F/5.0F*x6*y2*z + x4*y4*z - 2.0F/5.0F*x2*y6*z + (1.0F/70.0F)*y8*z) * dL_dRGB;
										dL_dsh[99] = SH_C9[18] * ((1.0F/126.0F)*x*(x2 - 3*y2)*(x6 - 33*x4*y2 + 27*x2*y4 - 3*y6)) * dL_dRGB;

										dRGBdx += SH_C9[0] * ((4.0F/7.0F)*x7*y - 4*x5*y3 + 4*x3*y5 - 4.0F/7.0F*x*y7) * sh[81] + SH_C9[1] * ((1.0F/7.0F)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[82] + SH_C9[2] * ((2.0F/85.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 1)) * sh[83] + SH_C9[3] * ((3.0F/170.0F)*y*z*(17*z2 - 3)*(5*x4 - 10*x2*y2 + y4)) * sh[84] + SH_C9[4] * ((2.0F/85.0F)*x*y*(x - y)*(x + y)*(85*z4 - 30*z2 + 1)) * sh[85] + SH_C9[5] * ((1.0F/17.0F)*y*z*(3*x2 - y2)*(17*z4 - 10*z2 + 1)) * sh[86] + SH_C9[6] * ((2.0F/221.0F)*x*y*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[87] + SH_C9[7] * ((1.0F/273.0F)*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[88] + SH_C9[10] * ((17.0F/28.0F)*z8 - z6 + (1.0F/2.0F)*z4 - 1.0F/13.0F*z2 + 1.0F/572.0F) * sh[91] + SH_C9[11] * ((2.0F/273.0F)*x*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[92] + SH_C9[12] * ((1.0F/221.0F)*(x - y)*(x + y)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[93] + SH_C9[13] * ((2.0F/51.0F)*x*z*(x2 - 3*y2)*(17*z4 - 10*z2 + 1)) * sh[94] + SH_C9[14] * ((1.0F/170.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(85*z4 - 30*z2 + 1)) * sh[95] + SH_C9[15] * ((2.0F/85.0F)*x*z*(17*z2 - 3)*(x4 - 10*x2*y2 + 5*y4)) * sh[96] + SH_C9[16] * ((1.0F/85.0F)*(x - y)*(x + y)*(17*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[97] + SH_C9[17] * ((4.0F/35.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[98] + SH_C9[18] * ((1.0F/14.0F)*x8 - 2*x6*y2 + 5*x4*y4 - 2*x2*y6 + (1.0F/14.0F)*y8) * sh[99];
										dRGBdy += SH_C9[0] * ((1.0F/14.0F)*x8 - 2*x6*y2 + 5*x4*y4 - 2*x2*y6 + (1.0F/14.0F)*y8) * sh[81] + SH_C9[1] * ((1.0F/7.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[82] + SH_C9[2] * ((1.0F/85.0F)*(x - y)*(x + y)*(17*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[83] + SH_C9[3] * ((3.0F/170.0F)*x*z*(17*z2 - 3)*(x4 - 10*x2*y2 + 5*y4)) * sh[84] + SH_C9[4] * ((1.0F/170.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(85*z4 - 30*z2 + 1)) * sh[85] + SH_C9[5] * ((1.0F/17.0F)*x*z*(x2 - 3*y2)*(17*z4 - 10*z2 + 1)) * sh[86] + SH_C9[6] * ((1.0F/221.0F)*(x - y)*(x + y)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[87] + SH_C9[7] * ((1.0F/273.0F)*x*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[88] + SH_C9[8] * ((17.0F/28.0F)*z8 - z6 + (1.0F/2.0F)*z4 - 1.0F/13.0F*z2 + 1.0F/572.0F) * sh[89] + SH_C9[11] * (-2.0F/273.0F*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[92] + SH_C9[12] * (-2.0F/221.0F*x*y*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[93] + SH_C9[13] * (-2.0F/51.0F*y*z*(3*x2 - y2)*(17*z4 - 10*z2 + 1)) * sh[94] + SH_C9[14] * (-2.0F/85.0F*x*y*(x - y)*(x + y)*(85*z4 - 30*z2 + 1)) * sh[95] + SH_C9[15] * (-2.0F/85.0F*y*z*(17*z2 - 3)*(5*x4 - 10*x2*y2 + y4)) * sh[96] + SH_C9[16] * (-2.0F/85.0F*x*y*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 1)) * sh[97] + SH_C9[17] * (-4.0F/35.0F*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[98] + SH_C9[18] * (-4.0F/7.0F*x7*y + 4*x5*y3 - 4*x3*y5 + (4.0F/7.0F)*x*y7) * sh[99];
										dRGBdz += SH_C9[1] * ((1.0F/7.0F)*x7*y - x5*y3 + x3*y5 - 1.0F/7.0F*x*y7) * sh[82] + SH_C9[2] * ((2.0F/35.0F)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[83] + SH_C9[3] * ((3.0F/170.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 1)) * sh[84] + SH_C9[4] * ((2.0F/85.0F)*y*z*(17*z2 - 3)*(5*x4 - 10*x2*y2 + y4)) * sh[85] + SH_C9[5] * ((1.0F/17.0F)*x*y*(x - y)*(x + y)*(85*z4 - 30*z2 + 1)) * sh[86] + SH_C9[6] * ((2.0F/17.0F)*y*z*(3*x2 - y2)*(17*z4 - 10*z2 + 1)) * sh[87] + SH_C9[7] * ((1.0F/39.0F)*x*y*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[88] + SH_C9[8] * ((2.0F/91.0F)*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[89] + SH_C9[9] * ((1.0F/572.0F)*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * sh[90] + SH_C9[10] * ((2.0F/91.0F)*x*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[91] + SH_C9[11] * ((1.0F/39.0F)*(x - y)*(x + y)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[92] + SH_C9[12] * ((2.0F/17.0F)*x*z*(x2 - 3*y2)*(17*z4 - 10*z2 + 1)) * sh[93] + SH_C9[13] * ((1.0F/102.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(85*z4 - 30*z2 + 1)) * sh[94] + SH_C9[14] * ((2.0F/85.0F)*x*z*(17*z2 - 3)*(x4 - 10*x2*y2 + 5*y4)) * sh[95] + SH_C9[15] * ((1.0F/85.0F)*(x - y)*(x + y)*(17*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[96] + SH_C9[16] * ((2.0F/35.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[97] + SH_C9[17] * ((1.0F/70.0F)*x8 - 2.0F/5.0F*x6*y2 + x4*y4 - 2.0F/5.0F*x2*y6 + (1.0F/70.0F)*y8) * sh[98];

									}
								}
							}
						}
					}
				}
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
	dL_ddir *= dir_multiplier;

	dL_drots[idx] += quat_to_rotmat_vjp(rots[idx], glm::outerProduct(raw_dir, dL_ddir));

	dL_ddir = rot * dL_ddir;

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float3* __restrict__ points_xy_image,
	const float4* __restrict__ normal_geovalue,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ out_others, 
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float4* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dgeovalue,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_geovalue[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	float T_back_face = 0;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	float dL_dback_alpha;
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		dL_dback_alpha = dL_depths[BACK_FACE_OFFSET * H * W + pix_id];
		T_back_face = out_others[BACK_FACE_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_geovalue[block.thread_rank()] = normal_geovalue[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float3 xym = collected_xy[j];
			const float2 xy = {xym.x, xym.y};
			const bool visible = xym.z > 0;
			// if (!visible) continue;
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (fabs(p.z) < 1E-8F) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d);

			// compute depth
			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z; // Tw * [u,v,1]
			// if a point is too small, its depth is not reliable?
			c_d = (rho3d <= rho2d) ? c_d : Tw.z; 
			if (c_d < near_n) continue;
			
			float4 nor_o = collected_normal_geovalue[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float geovalue = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, footprint_activation(geovalue * G));
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = (visible)? collected_colors[ch * BLOCK_SIZE + j] : 0.0F;
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				if (visible) atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			if ( !visible ) {
				dL_dalpha += dL_dback_alpha * (1.f - T_back_face) / (1.f - alpha);
			}

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			dL_dalpha *= dfootprint_activation(geovalue * G);
			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);

				atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dTu.z));
				atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dTv.z));
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				// // Propagate the gradients of depth
				// atomicAdd(&dL_dtransMat[global_id * 9 + 6],  s.x * dL_dz);
				// atomicAdd(&dL_dtransMat[global_id * 9 + 7],  s.y * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);
			}

			// Update gradients w.r.t. geovalue of the Gaussian
			atomicAdd(&(dL_dgeovalue[global_id]), G * dL_dalpha);
		}
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float4* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec2* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
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

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float4 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIBLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1])
	);
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float4* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, rotations, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, dL_drots, (glm::vec3*)dL_dshs);
	
	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
	dL_dmean2Ds[idx].z = fabs(dL_dmean2Ds[idx].z * depth * 0.5 * float(W)); // to ndc 
	dL_dmean2Ds[idx].w = fabs(dL_dmean2Ds[idx].w * depth * 0.5 * float(H)); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float4* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec2* dL_dscales,
	glm::vec4* dL_drots)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots
	);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float3* means2D,
	const float4* normal_geovalue,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* out_others, 
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float4* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dgeovalue,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_geovalue,
		transMats,
		colors,
		depths,
		final_Ts,
		n_contrib,
		out_others, 
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dgeovalue,
		dL_dcolors
		);
}
