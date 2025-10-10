#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <curand_kernel.h>

#define MAX_SH_DEGREE 9

__forceinline__ __device__ int* index_cached_form_factor(
    int* packed_matrix, 
    const unsigned int i, 
    const unsigned int j
) {
	if ( packed_matrix == nullptr ) return nullptr;
	long long N_NON_PL = (long long)(*packed_matrix + 0);
	// long long N_PL = (long long)(*packed_matrix + 1);
	if ( (long long)max(i, j) < N_NON_PL ) return nullptr;
	long long I = (long long)max(i, j) - N_NON_PL;
	long long J = (long long)min(i, j);
	return (packed_matrix + 2) + I * N_NON_PL + J;
}

__forceinline__ __device__ int read_cached_form_factor(
    int* packed_matrix, 
    const unsigned int i, 
    const unsigned int j
) {
    if ( packed_matrix == nullptr ) return -1;
	const auto ptr = index_cached_form_factor(packed_matrix, i, j);
	if ( ptr == nullptr ) return -1;
    return *ptr;
}

__forceinline__ __device__ void write_to_cached_form_factor(
    int* packed_matrix, 
    const unsigned int i, 
    const unsigned int j, 
	const int v
) {
    if ( packed_matrix == nullptr ) return;
	const auto ptr = index_cached_form_factor(packed_matrix, i, j);
	if ( ptr == nullptr ) return;
	*ptr = v;
}

__forceinline__ __device__ void atomicAdd(glm::vec3* addr, const glm::vec3 v) {
	atomicAdd(&(addr->x), v.x);
	atomicAdd(&(addr->y), v.y);
	atomicAdd(&(addr->z), v.z);
}

// Spherical harmonics coefficients
__device__ const float SH_C0[] = {0.282094791773878F};
__device__ const float SH_C1[] = {0.488602511902920F, 0.488602511902920F, 0.488602511902920F};
__device__ const float SH_C2[] = {1.09254843059208F, 1.09254843059208F, 0.946174695757560F, 1.09254843059208F, 0.546274215296040F};
__device__ const float SH_C3[] = {1.77013076977993F, 2.89061144264055F, 2.28522899732233F, 1.86588166295058F, 2.28522899732233F, 1.44530572132028F, 1.77013076977993F};
__device__ const float SH_C4[] = {2.50334294179671F, 5.31039230933979F, 6.62322287030292F, 4.68332580490102F, 3.70249414203215F, 4.68332580490102F, 3.31161143515146F, 5.31039230933979F, 3.75501441269506F};
__device__ const float SH_C5[] = {6.56382056840170F, 8.30264925952417F, 13.2094340847518F, 14.3806103549200F, 9.51187967510964F, 8.18652257173965F, 9.51187967510964F, 7.19030517745999F, 13.2094340847518F, 12.4539738892862F, 6.56382056840170F};
__device__ const float SH_C6[] = {13.6636821038383F, 23.6661916223175F, 22.2008556320639F, 30.3997735639925F, 30.3997735639925F, 19.2265049631181F, 20.0242987143030F, 19.2265049631181F, 15.1998867819962F, 30.3997735639925F, 33.3012834480958F, 23.6661916223175F, 10.2477615778787F};
__device__ const float SH_C7[] = {24.7506956383609F, 52.9192132360380F, 67.4590252336339F, 53.9672201869071F, 67.1208826269242F, 63.2821750196325F, 44.7141457533461F, 47.3210039000194F, 44.7141457533461F, 31.6410875098163F, 67.1208826269242F, 80.9508302803606F, 67.4590252336339F, 39.6894099270285F, 24.7506956383609F};
__device__ const float SH_C8[] = {40.8198929697905F, 102.049732424476F, 159.699829817863F, 172.495531104905F, 124.388296437426F, 144.526140169579F, 130.459545912384F, 109.150287144679F, 109.150287144679F, 109.150287144679F, 65.2297729561921F, 144.526140169579F, 186.582444656139F, 172.495531104905F, 119.774872363397F, 102.049732424476F, 51.0248662122381F};
__device__ const float SH_C9[] = {94.3615199335017F, 177.929788341463F, 324.218761399712F, 427.857803818173F, 414.271537183166F, 277.283548233584F, 306.112748770737F, 330.067021748918F, 258.025260101518F, 247.269437785311F, 258.025260101518F, 165.033510874459F, 306.112748770737F, 415.925322350376F, 414.271537183166F, 320.893352863630F, 324.218761399712F, 222.412235426829F, 94.3615199335017F};

__device__ glm::vec3 computeOutRadianceFromSH(
    const unsigned int deg, 
    const glm::vec3* sh, 
    const glm::vec3 local_frame_dir, 
    const bool clamp
) {
    // Convention: Always pointing outwards
	glm::vec3 dir = (local_frame_dir.z >= 0.0F)? local_frame_dir : -local_frame_dir;
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

    if ( clamp )
        result = glm::max(result, 0.0F);

	return result;
}

__device__ void computeSHResponse(
	const glm::vec3 coeff, 
    const unsigned int deg, 
	const glm::vec3* brdf_coeffs, 
    const glm::vec3 local_frame_dir, 
	glm::vec3* result
) {
    // Convention: Always pointing outwards
	glm::vec3 dir = (local_frame_dir.z >= 0.0F)? local_frame_dir : -local_frame_dir;
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	result[0] += coeff * SH_C0[0] * brdf_coeffs[0];
	if (deg > 0)
	{
		result[1] += coeff * SH_C1[0] * (y) * brdf_coeffs[1];
		result[2] += coeff * SH_C1[1] * (z) * brdf_coeffs[2];
		result[3] += coeff * SH_C1[2] * (x) * brdf_coeffs[3];
		if (deg > 1)
		{
			float x2 = x * x;
			float y2 = y * y;
			float z2 = z * z;
			result[4] += coeff * SH_C2[0] * (x*y) * brdf_coeffs[4];
			result[5] += coeff * SH_C2[1] * (y*z) * brdf_coeffs[5];
			result[6] += coeff * SH_C2[2] * (z2 - 1.0F/3.0F) * brdf_coeffs[6];
			result[7] += coeff * SH_C2[3] * (x*z) * brdf_coeffs[7];
			result[8] += coeff * SH_C2[4] * (x2 - y2) * brdf_coeffs[8];
			if (deg > 2)
			{
				float x3 = x2 * x;
				float y3 = y2 * y;
				float z3 = z2 * z;
				result[9] += coeff * SH_C3[0] * (x2*y - 1.0F/3.0F*y3) * brdf_coeffs[9];
				result[10] += coeff * SH_C3[1] * (x*y*z) * brdf_coeffs[10];
				result[11] += coeff * SH_C3[2] * (y*z2 - 1.0F/5.0F*y) * brdf_coeffs[11];
				result[12] += coeff * SH_C3[3] * (z3 - 3.0F/5.0F*z) * brdf_coeffs[12];
				result[13] += coeff * SH_C3[4] * (x*z2 - 1.0F/5.0F*x) * brdf_coeffs[13];
				result[14] += coeff * SH_C3[5] * (z*(x - y)*(x + y)) * brdf_coeffs[14];
				result[15] += coeff * SH_C3[6] * ((1.0F/3.0F)*x3 - x*y2) * brdf_coeffs[15];
				if (deg > 3)
				{
					float x4 = x3 * x;
					float y4 = y3 * y;
					float z4 = z3 * z;
					result[16] += coeff * SH_C4[0] * (x3*y - x*y3) * brdf_coeffs[16];
					result[17] += coeff * SH_C4[1] * (x2*y*z - 1.0F/3.0F*y3*z) * brdf_coeffs[17];
					result[18] += coeff * SH_C4[2] * (x*y*z2 - 1.0F/7.0F*x*y) * brdf_coeffs[18];
					result[19] += coeff * SH_C4[3] * (y*z3 - 3.0F/7.0F*y*z) * brdf_coeffs[19];
					result[20] += coeff * SH_C4[4] * (z4 - 6.0F/7.0F*z2 + 3.0F/35.0F) * brdf_coeffs[20];
					result[21] += coeff * SH_C4[5] * (x*z3 - 3.0F/7.0F*x*z) * brdf_coeffs[21];
					result[22] += coeff * SH_C4[6] * ((1.0F/7.0F)*(x - y)*(x + y)*(7*z2 - 1)) * brdf_coeffs[22];
					result[23] += coeff * SH_C4[7] * ((1.0F/3.0F)*x3*z - x*y2*z) * brdf_coeffs[23];
					result[24] += coeff * SH_C4[8] * ((1.0F/6.0F)*x4 - x2*y2 + (1.0F/6.0F)*y4) * brdf_coeffs[24];
					if (deg > 4)
					{
						float x5 = x4 * x;
						float y5 = y4 * y;
						float z5 = z4 * z;
						result[25] += coeff * SH_C5[0] * ((1.0F/2.0F)*x4*y - x2*y3 + (1.0F/10.0F)*y5) * brdf_coeffs[25];
						result[26] += coeff * SH_C5[1] * (x*y*z*(x - y)*(x + y)) * brdf_coeffs[26];
						result[27] += coeff * SH_C5[2] * ((1.0F/27.0F)*y*(3*x2 - y2)*(3*z - 1)*(3*z + 1)) * brdf_coeffs[27];
						result[28] += coeff * SH_C5[3] * (x*y*z3 - 1.0F/3.0F*x*y*z) * brdf_coeffs[28];
						result[29] += coeff * SH_C5[4] * ((1.0F/21.0F)*y*(21*z4 - 14*z2 + 1)) * brdf_coeffs[29];
						result[30] += coeff * SH_C5[5] * ((9.0F/10.0F)*z5 - z3 + (3.0F/14.0F)*z) * brdf_coeffs[30];
						result[31] += coeff * SH_C5[6] * ((1.0F/21.0F)*x*(21*z4 - 14*z2 + 1)) * brdf_coeffs[31];
						result[32] += coeff * SH_C5[7] * ((1.0F/3.0F)*z*(x - y)*(x + y)*(3*z2 - 1)) * brdf_coeffs[32];
						result[33] += coeff * SH_C5[8] * ((1.0F/27.0F)*x*(x2 - 3*y2)*(3*z - 1)*(3*z + 1)) * brdf_coeffs[33];
						result[34] += coeff * SH_C5[9] * ((1.0F/6.0F)*x4*z - x2*y2*z + (1.0F/6.0F)*y4*z) * brdf_coeffs[34];
						result[35] += coeff * SH_C5[10] * ((1.0F/10.0F)*x5 - x3*y2 + (1.0F/2.0F)*x*y4) * brdf_coeffs[35];
						if (deg > 5)
						{
							float x6 = x5 * x;
							float y6 = y5 * y;
							float z6 = z5 * z;
							result[36] += coeff * SH_C6[0] * ((1.0F/10.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * brdf_coeffs[36];
							result[37] += coeff * SH_C6[1] * ((1.0F/10.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * brdf_coeffs[37];
							result[38] += coeff * SH_C6[2] * ((1.0F/11.0F)*x*y*(x - y)*(x + y)*(11*z2 - 1)) * brdf_coeffs[38];
							result[39] += coeff * SH_C6[3] * ((1.0F/33.0F)*y*z*(3*x2 - y2)*(11*z2 - 3)) * brdf_coeffs[39];
							result[40] += coeff * SH_C6[4] * ((1.0F/33.0F)*x*y*(33*z4 - 18*z2 + 1)) * brdf_coeffs[40];
							result[41] += coeff * SH_C6[5] * ((1.0F/33.0F)*y*z*(33*z4 - 30*z2 + 5)) * brdf_coeffs[41];
							result[42] += coeff * SH_C6[6] * ((11.0F/15.0F)*z6 - z4 + (1.0F/3.0F)*z2 - 1.0F/63.0F) * brdf_coeffs[42];
							result[43] += coeff * SH_C6[7] * ((1.0F/33.0F)*x*z*(33*z4 - 30*z2 + 5)) * brdf_coeffs[43];
							result[44] += coeff * SH_C6[8] * ((1.0F/33.0F)*(x - y)*(x + y)*(33*z4 - 18*z2 + 1)) * brdf_coeffs[44];
							result[45] += coeff * SH_C6[9] * ((1.0F/33.0F)*x*z*(x2 - 3*y2)*(11*z2 - 3)) * brdf_coeffs[45];
							result[46] += coeff * SH_C6[10] * ((1.0F/66.0F)*(11*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * brdf_coeffs[46];
							result[47] += coeff * SH_C6[11] * ((1.0F/10.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * brdf_coeffs[47];
							result[48] += coeff * SH_C6[12] * ((1.0F/15.0F)*x6 - x4*y2 + x2*y4 - 1.0F/15.0F*y6) * brdf_coeffs[48];
							if (deg > 6)
							{
								float x7 = x6 * x;
								float y7 = y6 * y;
								float z7 = z6 * z;
								result[49] += coeff * SH_C7[0] * ((1.0F/5.0F)*x6*y - x4*y3 + (3.0F/5.0F)*x2*y5 - 1.0F/35.0F*y7) * brdf_coeffs[49];
								result[50] += coeff * SH_C7[1] * ((1.0F/10.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * brdf_coeffs[50];
								result[51] += coeff * SH_C7[2] * ((1.0F/130.0F)*y*(13*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * brdf_coeffs[51];
								result[52] += coeff * SH_C7[3] * ((1.0F/13.0F)*x*y*z*(x - y)*(x + y)*(13*z2 - 3)) * brdf_coeffs[52];
								result[53] += coeff * SH_C7[4] * ((1.0F/429.0F)*y*(3*x2 - y2)*(143*z4 - 66*z2 + 3)) * brdf_coeffs[53];
								result[54] += coeff * SH_C7[5] * ((1.0F/143.0F)*x*y*z*(143*z4 - 110*z2 + 15)) * brdf_coeffs[54];
								result[55] += coeff * SH_C7[6] * ((1.0F/495.0F)*y*(429*z6 - 495*z4 + 135*z2 - 5)) * brdf_coeffs[55];
								result[56] += coeff * SH_C7[7] * ((1.0F/693.0F)*z*(429*z6 - 693*z4 + 315*z2 - 35)) * brdf_coeffs[56];
								result[57] += coeff * SH_C7[8] * ((1.0F/495.0F)*x*(429*z6 - 495*z4 + 135*z2 - 5)) * brdf_coeffs[57];
								result[58] += coeff * SH_C7[9] * ((1.0F/143.0F)*z*(x - y)*(x + y)*(143*z4 - 110*z2 + 15)) * brdf_coeffs[58];
								result[59] += coeff * SH_C7[10] * ((1.0F/429.0F)*x*(x2 - 3*y2)*(143*z4 - 66*z2 + 3)) * brdf_coeffs[59];
								result[60] += coeff * SH_C7[11] * ((1.0F/78.0F)*z*(13*z2 - 3)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * brdf_coeffs[60];
								result[61] += coeff * SH_C7[12] * ((1.0F/130.0F)*x*(13*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * brdf_coeffs[61];
								result[62] += coeff * SH_C7[13] * ((1.0F/15.0F)*x6*z - x4*y2*z + x2*y4*z - 1.0F/15.0F*y6*z) * brdf_coeffs[62];
								result[63] += coeff * SH_C7[14] * ((1.0F/35.0F)*x7 - 3.0F/5.0F*x5*y2 + x3*y4 - 1.0F/5.0F*x*y6) * brdf_coeffs[63];
								if (deg > 7)
								{
									float x8 = x7 * x;
									float y8 = y7 * y;
									float z8 = z7 * z;
									result[64] += coeff * SH_C8[0] * ((1.0F/7.0F)*x7*y - x5*y3 + x3*y5 - 1.0F/7.0F*x*y7) * brdf_coeffs[64];
									result[65] += coeff * SH_C8[1] * ((1.0F/35.0F)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * brdf_coeffs[65];
									result[66] += coeff * SH_C8[2] * ((1.0F/150.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)*(15*z2 - 1)) * brdf_coeffs[66];
									result[67] += coeff * SH_C8[3] * ((1.0F/50.0F)*y*z*(5*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * brdf_coeffs[67];
									result[68] += coeff * SH_C8[4] * ((1.0F/65.0F)*x*y*(x - y)*(x + y)*(65*z4 - 26*z2 + 1)) * brdf_coeffs[68];
									result[69] += coeff * SH_C8[5] * ((1.0F/117.0F)*y*z*(3*x2 - y2)*(39*z4 - 26*z2 + 3)) * brdf_coeffs[69];
									result[70] += coeff * SH_C8[6] * ((1.0F/143.0F)*x*y*(143*z6 - 143*z4 + 33*z2 - 1)) * brdf_coeffs[70];
									result[71] += coeff * SH_C8[7] * ((1.0F/1001.0F)*y*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * brdf_coeffs[71];
									result[72] += coeff * SH_C8[8] * ((1.0F/12012.0F)*(6435*z8 - 12012*z6 + 6930*z4 - 1260*z2 + 35)) * brdf_coeffs[72];
									result[73] += coeff * SH_C8[9] * ((1.0F/1001.0F)*x*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * brdf_coeffs[73];
									result[74] += coeff * SH_C8[10] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z6 - 143*z4 + 33*z2 - 1)) * brdf_coeffs[74];
									result[75] += coeff * SH_C8[11] * ((1.0F/117.0F)*x*z*(x2 - 3*y2)*(39*z4 - 26*z2 + 3)) * brdf_coeffs[75];
									result[76] += coeff * SH_C8[12] * ((1.0F/390.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(65*z4 - 26*z2 + 1)) * brdf_coeffs[76];
									result[77] += coeff * SH_C8[13] * ((1.0F/50.0F)*x*z*(5*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * brdf_coeffs[77];
									result[78] += coeff * SH_C8[14] * ((1.0F/225.0F)*(x - y)*(x + y)*(15*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * brdf_coeffs[78];
									result[79] += coeff * SH_C8[15] * ((1.0F/35.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * brdf_coeffs[79];
									result[80] += coeff * SH_C8[16] * ((1.0F/70.0F)*x8 - 2.0F/5.0F*x6*y2 + x4*y4 - 2.0F/5.0F*x2*y6 + (1.0F/70.0F)*y8) * brdf_coeffs[80];
									if (deg > 8)
									{
										result[81] += coeff * SH_C9[0] * ((1.0F/126.0F)*y*(3*x2 - y2)*(3*x6 - 27*x4*y2 + 33*x2*y4 - y6)) * brdf_coeffs[81];
										result[82] += coeff * SH_C9[1] * ((1.0F/7.0F)*x7*y*z - x5*y3*z + x3*y5*z - 1.0F/7.0F*x*y7*z) * brdf_coeffs[82];
										result[83] += coeff * SH_C9[2] * ((1.0F/595.0F)*y*(17*z2 - 1)*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * brdf_coeffs[83];
										result[84] += coeff * SH_C9[3] * ((1.0F/170.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 3)) * brdf_coeffs[84];
										result[85] += coeff * SH_C9[4] * ((1.0F/850.0F)*y*(5*x4 - 10*x2*y2 + y4)*(85*z4 - 30*z2 + 1)) * brdf_coeffs[85];
										result[86] += coeff * SH_C9[5] * ((1.0F/17.0F)*x*y*z*(x - y)*(x + y)*(17*z4 - 10*z2 + 1)) * brdf_coeffs[86];
										result[87] += coeff * SH_C9[6] * ((1.0F/663.0F)*y*(3*x2 - y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * brdf_coeffs[87];
										result[88] += coeff * SH_C9[7] * ((1.0F/273.0F)*x*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * brdf_coeffs[88];
										result[89] += coeff * SH_C9[8] * ((1.0F/4004.0F)*y*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * brdf_coeffs[89];
										result[90] += coeff * SH_C9[9] * ((1.0F/25740.0F)*z*(12155*z8 - 25740*z6 + 18018*z4 - 4620*z2 + 315)) * brdf_coeffs[90];
										result[91] += coeff * SH_C9[10] * ((1.0F/4004.0F)*x*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * brdf_coeffs[91];
										result[92] += coeff * SH_C9[11] * ((1.0F/273.0F)*z*(x - y)*(x + y)*(221*z6 - 273*z4 + 91*z2 - 7)) * brdf_coeffs[92];
										result[93] += coeff * SH_C9[12] * ((1.0F/663.0F)*x*(x2 - 3*y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * brdf_coeffs[93];
										result[94] += coeff * SH_C9[13] * ((1.0F/102.0F)*z*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(17*z4 - 10*z2 + 1)) * brdf_coeffs[94];
										result[95] += coeff * SH_C9[14] * ((1.0F/850.0F)*x*(x4 - 10*x2*y2 + 5*y4)*(85*z4 - 30*z2 + 1)) * brdf_coeffs[95];
										result[96] += coeff * SH_C9[15] * ((1.0F/255.0F)*z*(x - y)*(x + y)*(17*z2 - 3)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * brdf_coeffs[96];
										result[97] += coeff * SH_C9[16] * ((1.0F/595.0F)*x*(17*z2 - 1)*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * brdf_coeffs[97];
										result[98] += coeff * SH_C9[17] * ((1.0F/70.0F)*x8*z - 2.0F/5.0F*x6*y2*z + x4*y4*z - 2.0F/5.0F*x2*y6*z + (1.0F/70.0F)*y8*z) * brdf_coeffs[98];
										result[99] += coeff * SH_C9[18] * ((1.0F/126.0F)*x*(x2 - 3*y2)*(x6 - 33*x4*y2 + 27*x2*y4 - 3*y6)) * brdf_coeffs[99];
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

__forceinline__ __device__ float square(const float x) {
	return x * x;
}

__forceinline__ __device__ glm::vec3 square(const glm::vec3 x) {
	return x * x;
}