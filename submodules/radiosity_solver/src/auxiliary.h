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

#define MAX_SH_STRIDE 100 // (9 + 1) ^ 2

__forceinline__ __device__ float square(const float v) {
	return v * v;
}

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

__forceinline__ __device__ glm::vec3 computeOutRadianceFromSH(
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

__forceinline__ __device__ float clampBaseProb(const int clamp_type, const glm::vec3 base_prob) {
    if ( clamp_type == 0 )
        return 0.2126F * abs(base_prob.x) + 0.7152F * abs(base_prob.y) + 0.0722F * abs(base_prob.z);
    else if ( clamp_type == 1 )
        return max(abs(base_prob.x), max(abs(base_prob.y), abs(base_prob.z)));
    else if ( clamp_type == 2 )
        return (1.0F / 3.0F) * (abs(base_prob.x) + abs(base_prob.y) + abs(base_prob.z));
    else if ( clamp_type == 3 )
        return glm::length(base_prob);
}

__forceinline__ __device__ float illumination_power_spectrum(const glm::vec3 x) {
	const auto y = clampBaseProb(0, x);
	return y * y;
}

__forceinline__ __device__ float computeSHResponsePower(
    const unsigned int deg, 
	const glm::vec3* sh, 
    const glm::vec3 local_frame_dir
) {
    // Convention: Always pointing outwards
	glm::vec3 dir = (local_frame_dir.z >= 0.0F)? local_frame_dir : -local_frame_dir;
	float result = 0.0F;
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	result += illumination_power_spectrum(SH_C0[0] * sh[0]);
	if (deg > 0)
	{
		result += illumination_power_spectrum(SH_C1[0] * (y) * sh[1]) + 
				illumination_power_spectrum(SH_C1[1] * (z) * sh[2]) + 
				illumination_power_spectrum(SH_C1[2] * (x) * sh[3]);
		if (deg > 1)
		{
			float x2 = x * x;
			float y2 = y * y;
			float z2 = z * z;
			result += illumination_power_spectrum(SH_C2[0] * (x*y) * sh[4]) + 
					illumination_power_spectrum(SH_C2[1] * (y*z) * sh[5]) + 
					illumination_power_spectrum(SH_C2[2] * (z2 - 1.0F/3.0F) * sh[6]) + 
					illumination_power_spectrum(SH_C2[3] * (x*z) * sh[7]) + 
					illumination_power_spectrum(SH_C2[4] * (x2 - y2) * sh[8]);
			if (deg > 2)
			{
				float x3 = x2 * x;
				float y3 = y2 * y;
				float z3 = z2 * z;
				result += illumination_power_spectrum(SH_C3[0] * (x2*y - 1.0F/3.0F*y3) * sh[9]) + 
						illumination_power_spectrum(SH_C3[1] * (x*y*z) * sh[10]) + 
						illumination_power_spectrum(SH_C3[2] * (y*z2 - 1.0F/5.0F*y) * sh[11]) + 
						illumination_power_spectrum(SH_C3[3] * (z3 - 3.0F/5.0F*z) * sh[12]) + 
						illumination_power_spectrum(SH_C3[4] * (x*z2 - 1.0F/5.0F*x) * sh[13]) + 
						illumination_power_spectrum(SH_C3[5] * (z*(x - y)*(x + y)) * sh[14]) + 
						illumination_power_spectrum(SH_C3[6] * ((1.0F/3.0F)*x3 - x*y2) * sh[15]);
				if (deg > 3)
				{
					float x4 = x3 * x;
					float y4 = y3 * y;
					float z4 = z3 * z;
					result += illumination_power_spectrum(SH_C4[0] * (x3*y - x*y3) * sh[16]) + 
							illumination_power_spectrum(SH_C4[1] * (x2*y*z - 1.0F/3.0F*y3*z) * sh[17]) + 
							illumination_power_spectrum(SH_C4[2] * (x*y*z2 - 1.0F/7.0F*x*y) * sh[18]) + 
							illumination_power_spectrum(SH_C4[3] * (y*z3 - 3.0F/7.0F*y*z) * sh[19]) + 
							illumination_power_spectrum(SH_C4[4] * (z4 - 6.0F/7.0F*z2 + 3.0F/35.0F) * sh[20]) + 
							illumination_power_spectrum(SH_C4[5] * (x*z3 - 3.0F/7.0F*x*z) * sh[21]) + 
							illumination_power_spectrum(SH_C4[6] * ((1.0F/7.0F)*(x - y)*(x + y)*(7*z2 - 1)) * sh[22]) + 
							illumination_power_spectrum(SH_C4[7] * ((1.0F/3.0F)*x3*z - x*y2*z) * sh[23]) + 
							illumination_power_spectrum(SH_C4[8] * ((1.0F/6.0F)*x4 - x2*y2 + (1.0F/6.0F)*y4) * sh[24]);
					if (deg > 4)
					{
						float x5 = x4 * x;
						float y5 = y4 * y;
						float z5 = z4 * z;
						result += illumination_power_spectrum(SH_C5[0] * ((1.0F/2.0F)*x4*y - x2*y3 + (1.0F/10.0F)*y5) * sh[25]) + 
								illumination_power_spectrum(SH_C5[1] * (x*y*z*(x - y)*(x + y)) * sh[26]) + 
								illumination_power_spectrum(SH_C5[2] * ((1.0F/27.0F)*y*(3*x2 - y2)*(3*z - 1)*(3*z + 1)) * sh[27]) + 
								illumination_power_spectrum(SH_C5[3] * (x*y*z3 - 1.0F/3.0F*x*y*z) * sh[28]) + 
								illumination_power_spectrum(SH_C5[4] * ((1.0F/21.0F)*y*(21*z4 - 14*z2 + 1)) * sh[29]) + 
								illumination_power_spectrum(SH_C5[5] * ((9.0F/10.0F)*z5 - z3 + (3.0F/14.0F)*z) * sh[30]) + 
								illumination_power_spectrum(SH_C5[6] * ((1.0F/21.0F)*x*(21*z4 - 14*z2 + 1)) * sh[31]) + 
								illumination_power_spectrum(SH_C5[7] * ((1.0F/3.0F)*z*(x - y)*(x + y)*(3*z2 - 1)) * sh[32]) + 
								illumination_power_spectrum(SH_C5[8] * ((1.0F/27.0F)*x*(x2 - 3*y2)*(3*z - 1)*(3*z + 1)) * sh[33]) + 
								illumination_power_spectrum(SH_C5[9] * ((1.0F/6.0F)*x4*z - x2*y2*z + (1.0F/6.0F)*y4*z) * sh[34]) + 
								illumination_power_spectrum(SH_C5[10] * ((1.0F/10.0F)*x5 - x3*y2 + (1.0F/2.0F)*x*y4) * sh[35]);
						if (deg > 5)
						{
							float x6 = x5 * x;
							float y6 = y5 * y;
							float z6 = z5 * z;
							result += illumination_power_spectrum(SH_C6[0] * ((1.0F/10.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * sh[36]) + 
									illumination_power_spectrum(SH_C6[1] * ((1.0F/10.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[37]) + 
									illumination_power_spectrum(SH_C6[2] * ((1.0F/11.0F)*x*y*(x - y)*(x + y)*(11*z2 - 1)) * sh[38]) + 
									illumination_power_spectrum(SH_C6[3] * ((1.0F/33.0F)*y*z*(3*x2 - y2)*(11*z2 - 3)) * sh[39]) + 
									illumination_power_spectrum(SH_C6[4] * ((1.0F/33.0F)*x*y*(33*z4 - 18*z2 + 1)) * sh[40]) + 
									illumination_power_spectrum(SH_C6[5] * ((1.0F/33.0F)*y*z*(33*z4 - 30*z2 + 5)) * sh[41]) + 
									illumination_power_spectrum(SH_C6[6] * ((11.0F/15.0F)*z6 - z4 + (1.0F/3.0F)*z2 - 1.0F/63.0F) * sh[42]) + 
									illumination_power_spectrum(SH_C6[7] * ((1.0F/33.0F)*x*z*(33*z4 - 30*z2 + 5)) * sh[43]) + 
									illumination_power_spectrum(SH_C6[8] * ((1.0F/33.0F)*(x - y)*(x + y)*(33*z4 - 18*z2 + 1)) * sh[44]) + 
									illumination_power_spectrum(SH_C6[9] * ((1.0F/33.0F)*x*z*(x2 - 3*y2)*(11*z2 - 3)) * sh[45]) + 
									illumination_power_spectrum(SH_C6[10] * ((1.0F/66.0F)*(11*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[46]) + 
									illumination_power_spectrum(SH_C6[11] * ((1.0F/10.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[47]) + 
									illumination_power_spectrum(SH_C6[12] * ((1.0F/15.0F)*x6 - x4*y2 + x2*y4 - 1.0F/15.0F*y6) * sh[48]);
							if (deg > 6)
							{
								float x7 = x6 * x;
								float y7 = y6 * y;
								float z7 = z6 * z;
								result += illumination_power_spectrum(SH_C7[0] * ((1.0F/5.0F)*x6*y - x4*y3 + (3.0F/5.0F)*x2*y5 - 1.0F/35.0F*y7) * sh[49]) + 
										illumination_power_spectrum(SH_C7[1] * ((1.0F/10.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[50]) + 
										illumination_power_spectrum(SH_C7[2] * ((1.0F/130.0F)*y*(13*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[51]) + 
										illumination_power_spectrum(SH_C7[3] * ((1.0F/13.0F)*x*y*z*(x - y)*(x + y)*(13*z2 - 3)) * sh[52]) + 
										illumination_power_spectrum(SH_C7[4] * ((1.0F/429.0F)*y*(3*x2 - y2)*(143*z4 - 66*z2 + 3)) * sh[53]) + 
										illumination_power_spectrum(SH_C7[5] * ((1.0F/143.0F)*x*y*z*(143*z4 - 110*z2 + 15)) * sh[54]) + 
										illumination_power_spectrum(SH_C7[6] * ((1.0F/495.0F)*y*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[55]) + 
										illumination_power_spectrum(SH_C7[7] * ((1.0F/693.0F)*z*(429*z6 - 693*z4 + 315*z2 - 35)) * sh[56]) + 
										illumination_power_spectrum(SH_C7[8] * ((1.0F/495.0F)*x*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[57]) + 
										illumination_power_spectrum(SH_C7[9] * ((1.0F/143.0F)*z*(x - y)*(x + y)*(143*z4 - 110*z2 + 15)) * sh[58]) + 
										illumination_power_spectrum(SH_C7[10] * ((1.0F/429.0F)*x*(x2 - 3*y2)*(143*z4 - 66*z2 + 3)) * sh[59]) + 
										illumination_power_spectrum(SH_C7[11] * ((1.0F/78.0F)*z*(13*z2 - 3)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[60]) + 
										illumination_power_spectrum(SH_C7[12] * ((1.0F/130.0F)*x*(13*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[61]) + 
										illumination_power_spectrum(SH_C7[13] * ((1.0F/15.0F)*x6*z - x4*y2*z + x2*y4*z - 1.0F/15.0F*y6*z) * sh[62]) + 
										illumination_power_spectrum(SH_C7[14] * ((1.0F/35.0F)*x7 - 3.0F/5.0F*x5*y2 + x3*y4 - 1.0F/5.0F*x*y6) * sh[63]);
								if (deg > 7)
								{
									float x8 = x7 * x;
									float y8 = y7 * y;
									float z8 = z7 * z;
									result += illumination_power_spectrum(SH_C8[0] * ((1.0F/7.0F)*x7*y - x5*y3 + x3*y5 - 1.0F/7.0F*x*y7) * sh[64]) + 
											illumination_power_spectrum(SH_C8[1] * ((1.0F/35.0F)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[65]) + 
											illumination_power_spectrum(SH_C8[2] * ((1.0F/150.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)*(15*z2 - 1)) * sh[66]) + 
											illumination_power_spectrum(SH_C8[3] * ((1.0F/50.0F)*y*z*(5*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[67]) + 
											illumination_power_spectrum(SH_C8[4] * ((1.0F/65.0F)*x*y*(x - y)*(x + y)*(65*z4 - 26*z2 + 1)) * sh[68]) + 
											illumination_power_spectrum(SH_C8[5] * ((1.0F/117.0F)*y*z*(3*x2 - y2)*(39*z4 - 26*z2 + 3)) * sh[69]) + 
											illumination_power_spectrum(SH_C8[6] * ((1.0F/143.0F)*x*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[70]) + 
											illumination_power_spectrum(SH_C8[7] * ((1.0F/1001.0F)*y*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[71]) + 
											illumination_power_spectrum(SH_C8[8] * ((1.0F/12012.0F)*(6435*z8 - 12012*z6 + 6930*z4 - 1260*z2 + 35)) * sh[72]) + 
											illumination_power_spectrum(SH_C8[9] * ((1.0F/1001.0F)*x*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[73]) + 
											illumination_power_spectrum(SH_C8[10] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[74]) + 
											illumination_power_spectrum(SH_C8[11] * ((1.0F/117.0F)*x*z*(x2 - 3*y2)*(39*z4 - 26*z2 + 3)) * sh[75]) + 
											illumination_power_spectrum(SH_C8[12] * ((1.0F/390.0F)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(65*z4 - 26*z2 + 1)) * sh[76]) + 
											illumination_power_spectrum(SH_C8[13] * ((1.0F/50.0F)*x*z*(5*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[77]) + 
											illumination_power_spectrum(SH_C8[14] * ((1.0F/225.0F)*(x - y)*(x + y)*(15*z2 - 1)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[78]) + 
											illumination_power_spectrum(SH_C8[15] * ((1.0F/35.0F)*x*z*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[79]) + 
											illumination_power_spectrum(SH_C8[16] * ((1.0F/70.0F)*x8 - 2.0F/5.0F*x6*y2 + x4*y4 - 2.0F/5.0F*x2*y6 + (1.0F/70.0F)*y8) * sh[80]);
									if (deg > 8)
									{
										result += illumination_power_spectrum(SH_C9[0] * ((1.0F/126.0F)*y*(3*x2 - y2)*(3*x6 - 27*x4*y2 + 33*x2*y4 - y6)) * sh[81]) + 
												illumination_power_spectrum(SH_C9[1] * ((1.0F/7.0F)*x7*y*z - x5*y3*z + x3*y5*z - 1.0F/7.0F*x*y7*z) * sh[82]) + 
												illumination_power_spectrum(SH_C9[2] * ((1.0F/595.0F)*y*(17*z2 - 1)*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[83]) + 
												illumination_power_spectrum(SH_C9[3] * ((1.0F/170.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 3)) * sh[84]) + 
												illumination_power_spectrum(SH_C9[4] * ((1.0F/850.0F)*y*(5*x4 - 10*x2*y2 + y4)*(85*z4 - 30*z2 + 1)) * sh[85]) + 
												illumination_power_spectrum(SH_C9[5] * ((1.0F/17.0F)*x*y*z*(x - y)*(x + y)*(17*z4 - 10*z2 + 1)) * sh[86]) + 
												illumination_power_spectrum(SH_C9[6] * ((1.0F/663.0F)*y*(3*x2 - y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[87]) + 
												illumination_power_spectrum(SH_C9[7] * ((1.0F/273.0F)*x*y*z*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[88]) + 
												illumination_power_spectrum(SH_C9[8] * ((1.0F/4004.0F)*y*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * sh[89]) + 
												illumination_power_spectrum(SH_C9[9] * ((1.0F/25740.0F)*z*(12155*z8 - 25740*z6 + 18018*z4 - 4620*z2 + 315)) * sh[90]) + 
												illumination_power_spectrum(SH_C9[10] * ((1.0F/4004.0F)*x*(2431*z8 - 4004*z6 + 2002*z4 - 308*z2 + 7)) * sh[91]) + 
												illumination_power_spectrum(SH_C9[11] * ((1.0F/273.0F)*z*(x - y)*(x + y)*(221*z6 - 273*z4 + 91*z2 - 7)) * sh[92]) + 
												illumination_power_spectrum(SH_C9[12] * ((1.0F/663.0F)*x*(x2 - 3*y2)*(221*z6 - 195*z4 + 39*z2 - 1)) * sh[93]) + 
												illumination_power_spectrum(SH_C9[13] * ((1.0F/102.0F)*z*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)*(17*z4 - 10*z2 + 1)) * sh[94]) + 
												illumination_power_spectrum(SH_C9[14] * ((1.0F/850.0F)*x*(x4 - 10*x2*y2 + 5*y4)*(85*z4 - 30*z2 + 1)) * sh[95]) + 
												illumination_power_spectrum(SH_C9[15] * ((1.0F/255.0F)*z*(x - y)*(x + y)*(17*z2 - 3)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[96]) + 
												illumination_power_spectrum(SH_C9[16] * ((1.0F/595.0F)*x*(17*z2 - 1)*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[97]) + 
												illumination_power_spectrum(SH_C9[17] * ((1.0F/70.0F)*x8*z - 2.0F/5.0F*x6*y2*z + x4*y4*z - 2.0F/5.0F*x2*y6*z + (1.0F/70.0F)*y8*z) * sh[98]) + 
												illumination_power_spectrum(SH_C9[18] * ((1.0F/126.0F)*x*(x2 - 3*y2)*(x6 - 33*x4*y2 + 27*x2*y4 - 3*y6)) * sh[99]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return sqrtf(result);
}

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

__forceinline__ __device__ glm::vec4 convert_normal_into_quaternion(const glm::vec3& normal) {
	return glm::normalize(glm::vec4(glm::length(normal) + normal.z, -normal.y, normal.x, 0.0F));
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

__forceinline__ __device__ glm::vec3 grad_sh_2_grad_dir(
	const unsigned int deg, 
	const glm::vec3 c, 
	const glm::vec3* sh, 
	const glm::vec3 local_frame_dir
) {
	glm::vec3 dir = local_frame_dir;
	float dir_multiplier = 1.0F;
	if ( dir.z < 0.0F ) dir_multiplier = -1.0F;
	dir = dir_multiplier * dir;
	
	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	if (deg > 0)
	{
		dRGBdx += SH_C1[2] * sh[3];
		dRGBdy += SH_C1[0] * sh[1];
		dRGBdz += SH_C1[1] * sh[2];

		if (deg > 1)
		{
			float x2 = x * x;
			float y2 = y * y;
			float z2 = z * z;
			dRGBdx += SH_C2[0] * (y) * sh[4] + SH_C2[3] * (z) * sh[7] + SH_C2[4] * (2*x) * sh[8];
			dRGBdy += SH_C2[0] * (x) * sh[4] + SH_C2[1] * (z) * sh[5] + SH_C2[4] * (-2*y) * sh[8];
			dRGBdz += SH_C2[1] * (y) * sh[5] + SH_C2[2] * (2*z) * sh[6] + SH_C2[3] * (x) * sh[7];

			if (deg > 2)
			{
				float x3 = x2 * x;
				float y3 = y2 * y;
				float z3 = z2 * z;
				dRGBdx += SH_C3[0] * (2*x*y) * sh[9] + SH_C3[1] * (y*z) * sh[10] + SH_C3[4] * (z2 - 1.0F/5.0F) * sh[13] + SH_C3[5] * (2*x*z) * sh[14] + SH_C3[6] * (x2 - y2) * sh[15];
				dRGBdy += SH_C3[0] * (x2 - y2) * sh[9] + SH_C3[1] * (x*z) * sh[10] + SH_C3[2] * (z2 - 1.0F/5.0F) * sh[11] + SH_C3[5] * (-2*y*z) * sh[14] + SH_C3[6] * (-2*x*y) * sh[15];
				dRGBdz += SH_C3[1] * (x*y) * sh[10] + SH_C3[2] * (2*y*z) * sh[11] + SH_C3[3] * (3*z2 - 3.0F/5.0F) * sh[12] + SH_C3[4] * (2*x*z) * sh[13] + SH_C3[5] * (x2 - y2) * sh[14];

				if (deg > 3)
				{
					float x4 = x3 * x;
					float y4 = y3 * y;
					float z4 = z3 * z;
					dRGBdx += SH_C4[0] * (3*x2*y - y3) * sh[16] + SH_C4[1] * (2*x*y*z) * sh[17] + SH_C4[2] * (y*z2 - 1.0F/7.0F*y) * sh[18] + SH_C4[5] * (z3 - 3.0F/7.0F*z) * sh[21] + SH_C4[6] * (2*x*z2 - 2.0F/7.0F*x) * sh[22] + SH_C4[7] * (z*(x - y)*(x + y)) * sh[23] + SH_C4[8] * ((2.0F/3.0F)*x3 - 2*x*y2) * sh[24];
					dRGBdy += SH_C4[0] * (x3 - 3*x*y2) * sh[16] + SH_C4[1] * (z*(x - y)*(x + y)) * sh[17] + SH_C4[2] * (x*z2 - 1.0F/7.0F*x) * sh[18] + SH_C4[3] * (z3 - 3.0F/7.0F*z) * sh[19] + SH_C4[6] * (-2*y*z2 + (2.0F/7.0F)*y) * sh[22] + SH_C4[7] * (-2*x*y*z) * sh[23] + SH_C4[8] * (-2*x2*y + (2.0F/3.0F)*y3) * sh[24];
					dRGBdz += SH_C4[1] * (x2*y - 1.0F/3.0F*y3) * sh[17] + SH_C4[2] * (2*x*y*z) * sh[18] + SH_C4[3] * (3*y*z2 - 3.0F/7.0F*y) * sh[19] + SH_C4[4] * (4*z3 - 12.0F/7.0F*z) * sh[20] + SH_C4[5] * (3*x*z2 - 3.0F/7.0F*x) * sh[21] + SH_C4[6] * (2*z*(x - y)*(x + y)) * sh[22] + SH_C4[7] * ((1.0F/3.0F)*x3 - x*y2) * sh[23];

					if (deg > 4)
					{
						float x5 = x4 * x;
						float y5 = y4 * y;
						float z5 = z4 * z;
						dRGBdx += SH_C5[0] * (2*x*y*(x - y)*(x + y)) * sh[25] + SH_C5[1] * (y*z*(3*x2 - y2)) * sh[26] + SH_C5[2] * (2*x*y*z2 - 2.0F/9.0F*x*y) * sh[27] + SH_C5[3] * (y*z3 - 1.0F/3.0F*y*z) * sh[28] + SH_C5[6] * (z4 - 2.0F/3.0F*z2 + 1.0F/21.0F) * sh[31] + SH_C5[7] * (2*x*z3 - 2.0F/3.0F*x*z) * sh[32] + SH_C5[8] * ((1.0F/9.0F)*(x - y)*(x + y)*(3*z - 1)*(3*z + 1)) * sh[33] + SH_C5[9] * ((2.0F/3.0F)*x*z*(x2 - 3*y2)) * sh[34] + SH_C5[10] * ((1.0F/2.0F)*x4 - 3*x2*y2 + (1.0F/2.0F)*y4) * sh[35];
						dRGBdy += SH_C5[0] * ((1.0F/2.0F)*x4 - 3*x2*y2 + (1.0F/2.0F)*y4) * sh[25] + SH_C5[1] * (x*z*(x2 - 3*y2)) * sh[26] + SH_C5[2] * ((1.0F/9.0F)*(x - y)*(x + y)*(3*z - 1)*(3*z + 1)) * sh[27] + SH_C5[3] * (x*z3 - 1.0F/3.0F*x*z) * sh[28] + SH_C5[4] * (z4 - 2.0F/3.0F*z2 + 1.0F/21.0F) * sh[29] + SH_C5[7] * (-2*y*z3 + (2.0F/3.0F)*y*z) * sh[32] + SH_C5[8] * (-2*x*y*z2 + (2.0F/9.0F)*x*y) * sh[33] + SH_C5[9] * (-2*x2*y*z + (2.0F/3.0F)*y3*z) * sh[34] + SH_C5[10] * (-2*x3*y + 2*x*y3) * sh[35];
						dRGBdz += SH_C5[1] * (x3*y - x*y3) * sh[26] + SH_C5[2] * ((2.0F/3.0F)*y*z*(3*x2 - y2)) * sh[27] + SH_C5[3] * (3*x*y*z2 - 1.0F/3.0F*x*y) * sh[28] + SH_C5[4] * (4*y*z3 - 4.0F/3.0F*y*z) * sh[29] + SH_C5[5] * ((3.0F/14.0F)*(21*z4 - 14*z2 + 1)) * sh[30] + SH_C5[6] * (4*x*z3 - 4.0F/3.0F*x*z) * sh[31] + SH_C5[7] * ((1.0F/3.0F)*(x - y)*(x + y)*(3*z - 1)*(3*z + 1)) * sh[32] + SH_C5[8] * ((2.0F/3.0F)*x*z*(x2 - 3*y2)) * sh[33] + SH_C5[9] * ((1.0F/6.0F)*x4 - x2*y2 + (1.0F/6.0F)*y4) * sh[34];

						if (deg > 5)
						{
							float x6 = x5 * x;
							float y6 = y5 * y;
							float z6 = z5 * z;
							dRGBdx += SH_C6[0] * ((3.0F/10.0F)*y*(5*x4 - 10*x2*y2 + y4)) * sh[36] + SH_C6[1] * (2*x*y*z*(x - y)*(x + y)) * sh[37] + SH_C6[2] * ((1.0F/11.0F)*y*(3*x2 - y2)*(11*z2 - 1)) * sh[38] + SH_C6[3] * ((2.0F/11.0F)*x*y*z*(11*z2 - 3)) * sh[39] + SH_C6[4] * ((1.0F/33.0F)*y*(33*z4 - 18*z2 + 1)) * sh[40] + SH_C6[7] * (z5 - 10.0F/11.0F*z3 + (5.0F/33.0F)*z) * sh[43] + SH_C6[8] * ((2.0F/33.0F)*x*(33*z4 - 18*z2 + 1)) * sh[44] + SH_C6[9] * ((1.0F/11.0F)*z*(x - y)*(x + y)*(11*z2 - 3)) * sh[45] + SH_C6[10] * ((2.0F/33.0F)*x*(x2 - 3*y2)*(11*z2 - 1)) * sh[46] + SH_C6[11] * ((1.0F/2.0F)*x4*z - 3*x2*y2*z + (1.0F/2.0F)*y4*z) * sh[47] + SH_C6[12] * ((2.0F/5.0F)*x5 - 4*x3*y2 + 2*x*y4) * sh[48];
							dRGBdy += SH_C6[0] * ((3.0F/10.0F)*x*(x4 - 10*x2*y2 + 5*y4)) * sh[36] + SH_C6[1] * ((1.0F/2.0F)*x4*z - 3*x2*y2*z + (1.0F/2.0F)*y4*z) * sh[37] + SH_C6[2] * ((1.0F/11.0F)*x*(x2 - 3*y2)*(11*z2 - 1)) * sh[38] + SH_C6[3] * ((1.0F/11.0F)*z*(x - y)*(x + y)*(11*z2 - 3)) * sh[39] + SH_C6[4] * ((1.0F/33.0F)*x*(33*z4 - 18*z2 + 1)) * sh[40] + SH_C6[5] * (z5 - 10.0F/11.0F*z3 + (5.0F/33.0F)*z) * sh[41] + SH_C6[8] * (-2.0F/33.0F*y*(33*z4 - 18*z2 + 1)) * sh[44] + SH_C6[9] * (-2*x*y*z3 + (6.0F/11.0F)*x*y*z) * sh[45] + SH_C6[10] * (-2.0F/33.0F*y*(3*x2 - y2)*(11*z2 - 1)) * sh[46] + SH_C6[11] * (-2*x*y*z*(x - y)*(x + y)) * sh[47] + SH_C6[12] * (-2*x4*y + 4*x2*y3 - 2.0F/5.0F*y5) * sh[48];
							dRGBdz += SH_C6[1] * ((1.0F/2.0F)*x4*y - x2*y3 + (1.0F/10.0F)*y5) * sh[37] + SH_C6[2] * (2*x*y*z*(x - y)*(x + y)) * sh[38] + SH_C6[3] * ((1.0F/11.0F)*y*(3*x2 - y2)*(11*z2 - 1)) * sh[39] + SH_C6[4] * ((4.0F/11.0F)*x*y*z*(11*z2 - 3)) * sh[40] + SH_C6[5] * ((5.0F/33.0F)*y*(33*z4 - 18*z2 + 1)) * sh[41] + SH_C6[6] * ((22.0F/5.0F)*z5 - 4*z3 + (2.0F/3.0F)*z) * sh[42] + SH_C6[7] * ((5.0F/33.0F)*x*(33*z4 - 18*z2 + 1)) * sh[43] + SH_C6[8] * ((4.0F/11.0F)*z*(x - y)*(x + y)*(11*z2 - 3)) * sh[44] + SH_C6[9] * ((1.0F/11.0F)*x*(x2 - 3*y2)*(11*z2 - 1)) * sh[45] + SH_C6[10] * ((1.0F/3.0F)*x4*z - 2*x2*y2*z + (1.0F/3.0F)*y4*z) * sh[46] + SH_C6[11] * ((1.0F/10.0F)*x5 - x3*y2 + (1.0F/2.0F)*x*y4) * sh[47];

							if (deg > 6)
							{
								float x7 = x6 * x;
								float y7 = y6 * y;
								float z7 = z6 * z;
								dRGBdx += SH_C7[0] * ((2.0F/5.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * sh[49] + SH_C7[1] * ((3.0F/10.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[50] + SH_C7[2] * ((2.0F/13.0F)*x*y*(x - y)*(x + y)*(13*z2 - 1)) * sh[51] + SH_C7[3] * ((1.0F/13.0F)*y*z*(3*x2 - y2)*(13*z2 - 3)) * sh[52] + SH_C7[4] * ((2.0F/143.0F)*x*y*(143*z4 - 66*z2 + 3)) * sh[53] + SH_C7[5] * ((1.0F/143.0F)*y*z*(143*z4 - 110*z2 + 15)) * sh[54] + SH_C7[8] * ((1.0F/495.0F)*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[57] + SH_C7[9] * ((2.0F/143.0F)*x*z*(143*z4 - 110*z2 + 15)) * sh[58] + SH_C7[10] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z4 - 66*z2 + 3)) * sh[59] + SH_C7[11] * ((2.0F/39.0F)*x*z*(x2 - 3*y2)*(13*z2 - 3)) * sh[60] + SH_C7[12] * ((1.0F/26.0F)*(13*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[61] + SH_C7[13] * ((2.0F/5.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[62] + SH_C7[14] * ((1.0F/5.0F)*x6 - 3*x4*y2 + 3*x2*y4 - 1.0F/5.0F*y6) * sh[63];
								dRGBdy += SH_C7[0] * ((1.0F/5.0F)*x6 - 3*x4*y2 + 3*x2*y4 - 1.0F/5.0F*y6) * sh[49] + SH_C7[1] * ((3.0F/10.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[50] + SH_C7[2] * ((1.0F/26.0F)*(13*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[51] + SH_C7[3] * ((1.0F/13.0F)*x*z*(x2 - 3*y2)*(13*z2 - 3)) * sh[52] + SH_C7[4] * ((1.0F/143.0F)*(x - y)*(x + y)*(143*z4 - 66*z2 + 3)) * sh[53] + SH_C7[5] * ((1.0F/143.0F)*x*z*(143*z4 - 110*z2 + 15)) * sh[54] + SH_C7[6] * ((1.0F/495.0F)*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[55] + SH_C7[9] * (-2.0F/143.0F*y*z*(143*z4 - 110*z2 + 15)) * sh[58] + SH_C7[10] * (-2.0F/143.0F*x*y*(143*z4 - 66*z2 + 3)) * sh[59] + SH_C7[11] * (-2.0F/39.0F*y*z*(3*x2 - y2)*(13*z2 - 3)) * sh[60] + SH_C7[12] * (-2.0F/13.0F*x*y*(x - y)*(x + y)*(13*z2 - 1)) * sh[61] + SH_C7[13] * (-2.0F/5.0F*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[62] + SH_C7[14] * (-6.0F/5.0F*x5*y + 4*x3*y3 - 6.0F/5.0F*x*y5) * sh[63];
								dRGBdz += SH_C7[1] * ((1.0F/10.0F)*x*y*(x2 - 3*y2)*(3*x2 - y2)) * sh[50] + SH_C7[2] * ((1.0F/5.0F)*y*z*(5*x4 - 10*x2*y2 + y4)) * sh[51] + SH_C7[3] * ((3.0F/13.0F)*x*y*(x - y)*(x + y)*(13*z2 - 1)) * sh[52] + SH_C7[4] * ((4.0F/39.0F)*y*z*(3*x2 - y2)*(13*z2 - 3)) * sh[53] + SH_C7[5] * ((5.0F/143.0F)*x*y*(143*z4 - 66*z2 + 3)) * sh[54] + SH_C7[6] * ((2.0F/55.0F)*y*z*(143*z4 - 110*z2 + 15)) * sh[55] + SH_C7[7] * ((1.0F/99.0F)*(429*z6 - 495*z4 + 135*z2 - 5)) * sh[56] + SH_C7[8] * ((2.0F/55.0F)*x*z*(143*z4 - 110*z2 + 15)) * sh[57] + SH_C7[9] * ((5.0F/143.0F)*(x - y)*(x + y)*(143*z4 - 66*z2 + 3)) * sh[58] + SH_C7[10] * ((4.0F/39.0F)*x*z*(x2 - 3*y2)*(13*z2 - 3)) * sh[59] + SH_C7[11] * ((1.0F/26.0F)*(13*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[60] + SH_C7[12] * ((1.0F/5.0F)*x*z*(x4 - 10*x2*y2 + 5*y4)) * sh[61] + SH_C7[13] * ((1.0F/15.0F)*x6 - x4*y2 + x2*y4 - 1.0F/15.0F*y6) * sh[62];

								if (deg > 7)
								{
									float x8 = x7 * x;
									float y8 = y7 * y;
									float z8 = z7 * z;
									dRGBdx += SH_C8[0] * (x6*y - 5*x4*y3 + 3*x2*y5 - 1.0F/7.0F*y7) * sh[64] + SH_C8[1] * ((2.0F/5.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[65] + SH_C8[2] * ((1.0F/50.0F)*y*(15*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[66] + SH_C8[3] * ((2.0F/5.0F)*x*y*z*(x - y)*(x + y)*(5*z2 - 1)) * sh[67] + SH_C8[4] * ((1.0F/65.0F)*y*(3*x2 - y2)*(65*z4 - 26*z2 + 1)) * sh[68] + SH_C8[5] * ((2.0F/39.0F)*x*y*z*(39*z4 - 26*z2 + 3)) * sh[69] + SH_C8[6] * ((1.0F/143.0F)*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[70] + SH_C8[9] * ((1.0F/1001.0F)*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[73] + SH_C8[10] * ((2.0F/143.0F)*x*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[74] + SH_C8[11] * ((1.0F/39.0F)*z*(x - y)*(x + y)*(39*z4 - 26*z2 + 3)) * sh[75] + SH_C8[12] * ((2.0F/195.0F)*x*(x2 - 3*y2)*(65*z4 - 26*z2 + 1)) * sh[76] + SH_C8[13] * ((1.0F/10.0F)*z*(5*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[77] + SH_C8[14] * ((2.0F/75.0F)*x*(15*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[78] + SH_C8[15] * ((1.0F/5.0F)*x6*z - 3*x4*y2*z + 3*x2*y4*z - 1.0F/5.0F*y6*z) * sh[79] + SH_C8[16] * ((4.0F/35.0F)*x*(x6 - 21*x4*y2 + 35*x2*y4 - 7*y6)) * sh[80];
									dRGBdy += SH_C8[0] * ((1.0F/7.0F)*x7 - 3*x5*y2 + 5*x3*y4 - x*y6) * sh[64] + SH_C8[1] * ((1.0F/5.0F)*x6*z - 3*x4*y2*z + 3*x2*y4*z - 1.0F/5.0F*y6*z) * sh[65] + SH_C8[2] * ((1.0F/50.0F)*x*(15*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[66] + SH_C8[3] * ((1.0F/10.0F)*z*(5*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[67] + SH_C8[4] * ((1.0F/65.0F)*x*(x2 - 3*y2)*(65*z4 - 26*z2 + 1)) * sh[68] + SH_C8[5] * ((1.0F/39.0F)*z*(x - y)*(x + y)*(39*z4 - 26*z2 + 3)) * sh[69] + SH_C8[6] * ((1.0F/143.0F)*x*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[70] + SH_C8[7] * ((1.0F/1001.0F)*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[71] + SH_C8[10] * (-2.0F/143.0F*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[74] + SH_C8[11] * (-2.0F/39.0F*x*y*z*(39*z4 - 26*z2 + 3)) * sh[75] + SH_C8[12] * (-2.0F/195.0F*y*(3*x2 - y2)*(65*z4 - 26*z2 + 1)) * sh[76] + SH_C8[13] * (-2.0F/5.0F*x*y*z*(x - y)*(x + y)*(5*z2 - 1)) * sh[77] + SH_C8[14] * (-2.0F/75.0F*y*(15*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[78] + SH_C8[15] * (-2.0F/5.0F*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[79] + SH_C8[16] * (-4.0F/35.0F*y*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6)) * sh[80];
									dRGBdz += SH_C8[1] * ((1.0F/5.0F)*x6*y - x4*y3 + (3.0F/5.0F)*x2*y5 - 1.0F/35.0F*y7) * sh[65] + SH_C8[2] * ((1.0F/5.0F)*x*y*z*(x2 - 3*y2)*(3*x2 - y2)) * sh[66] + SH_C8[3] * ((1.0F/50.0F)*y*(15*z2 - 1)*(5*x4 - 10*x2*y2 + y4)) * sh[67] + SH_C8[4] * ((4.0F/5.0F)*x*y*z*(x - y)*(x + y)*(5*z2 - 1)) * sh[68] + SH_C8[5] * ((1.0F/39.0F)*y*(3*x2 - y2)*(65*z4 - 26*z2 + 1)) * sh[69] + SH_C8[6] * ((2.0F/13.0F)*x*y*z*(39*z4 - 26*z2 + 3)) * sh[70] + SH_C8[7] * ((5.0F/143.0F)*y*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[71] + SH_C8[8] * ((6.0F/1001.0F)*z*(715*z6 - 1001*z4 + 385*z2 - 35)) * sh[72] + SH_C8[9] * ((5.0F/143.0F)*x*(143*z6 - 143*z4 + 33*z2 - 1)) * sh[73] + SH_C8[10] * ((2.0F/13.0F)*z*(x - y)*(x + y)*(39*z4 - 26*z2 + 3)) * sh[74] + SH_C8[11] * ((1.0F/39.0F)*x*(x2 - 3*y2)*(65*z4 - 26*z2 + 1)) * sh[75] + SH_C8[12] * ((2.0F/15.0F)*z*(5*z2 - 1)*(x2 - 2*x*y - y2)*(x2 + 2*x*y - y2)) * sh[76] + SH_C8[13] * ((1.0F/50.0F)*x*(15*z2 - 1)*(x4 - 10*x2*y2 + 5*y4)) * sh[77] + SH_C8[14] * ((2.0F/15.0F)*z*(x - y)*(x + y)*(x2 - 4*x*y + y2)*(x2 + 4*x*y + y2)) * sh[78] + SH_C8[15] * ((1.0F/35.0F)*x7 - 3.0F/5.0F*x5*y2 + x3*y4 - 1.0F/5.0F*x*y6) * sh[79];

									if (deg > 8)
									{
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
	glm::vec3 dL_ddir(glm::dot(dRGBdx, c), glm::dot(dRGBdy, c), glm::dot(dRGBdz, c));
	dL_ddir *= dir_multiplier;
	return dL_ddir;
}