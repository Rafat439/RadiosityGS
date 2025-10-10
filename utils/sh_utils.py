#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = [0.282094791773878]
C1 = [0.488602511902920, 0.488602511902920, 0.488602511902920]
C2 = [1.09254843059208, 1.09254843059208, 0.946174695757560, 1.09254843059208, 0.546274215296040]
C3 = [1.77013076977993, 2.89061144264055, 2.28522899732233, 1.86588166295058, 2.28522899732233, 1.44530572132028, 1.77013076977993]
C4 = [2.50334294179671, 5.31039230933979, 6.62322287030292, 4.68332580490102, 3.70249414203215, 4.68332580490102, 3.31161143515146, 5.31039230933979, 3.75501441269506]
C5 = [6.56382056840170, 8.30264925952417, 13.2094340847518, 14.3806103549200, 9.51187967510964, 8.18652257173965, 9.51187967510964, 7.19030517745999, 13.2094340847518, 12.4539738892862, 6.56382056840170]
C6 = [13.6636821038383, 23.6661916223175, 22.2008556320639, 30.3997735639925, 30.3997735639925, 19.2265049631181, 20.0242987143030, 19.2265049631181, 15.1998867819962, 30.3997735639925, 33.3012834480958, 23.6661916223175, 10.2477615778787]
C7 = [24.7506956383609, 52.9192132360380, 67.4590252336339, 53.9672201869071, 67.1208826269242, 63.2821750196325, 44.7141457533461, 47.3210039000194, 44.7141457533461, 31.6410875098163, 67.1208826269242, 80.9508302803606, 67.4590252336339, 39.6894099270285, 24.7506956383609]
C8 = [40.8198929697905, 102.049732424476, 159.699829817863, 172.495531104905, 124.388296437426, 144.526140169579, 130.459545912384, 109.150287144679, 109.150287144679, 109.150287144679, 65.2297729561921, 144.526140169579, 186.582444656139, 172.495531104905, 119.774872363397, 102.049732424476, 51.0248662122381]
C9 = [94.3615199335017, 177.929788341463, 324.218761399712, 427.857803818173, 414.271537183166, 277.283548233584, 306.112748770737, 330.067021748918, 258.025260101518, 247.269437785311, 258.025260101518, 165.033510874459, 306.112748770737, 415.925322350376, 414.271537183166, 320.893352863630, 324.218761399712, 222.412235426829, 94.3615199335017]

def eval_sh_response(deg, coeff, dirs):
    assert deg <= 9 and deg >= 0
    result = []

    dirs = torch.where(dirs[..., 2:3] > 0, dirs, -dirs)

    result.append(coeff * C0[0] * torch.ones_like(dirs[..., 0:1]))
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result.append(coeff * C1[0] * (y))
        result.append(coeff * C1[1] * (z))
        result.append(coeff * C1[2] * (x))
        if deg > 1:
            x2, y2, z2 = x * x, y * y, z * z
            result.append(coeff * C2[0] * (x*y))
            result.append(coeff * C2[1] * (y*z))
            result.append(coeff * C2[2] * (z2 - 1/3))
            result.append(coeff * C2[3] * (x*z))
            result.append(coeff * C2[4] * (x2 - y2))
            if deg > 2:
                x3, y3, z3 = x2 * x, y2 * y, z2 * z
                result.append(coeff * C3[0] * (x2*y - 1/3*y3))
                result.append(coeff * C3[1] * (x*y*z))
                result.append(coeff * C3[2] * (y*z2 - 1/5*y))
                result.append(coeff * C3[3] * (z3 - 3/5*z))
                result.append(coeff * C3[4] * (x*z2 - 1/5*x))
                result.append(coeff * C3[5] * (x2*z - y2*z))
                result.append(coeff * C3[6] * ((1/3)*x3 - x*y2))
                if deg > 3:
                    x4, y4, z4 = x3 * x, y3 * y, z3 * z
                    result.append(coeff * C4[0] * (x3*y - x*y3))
                    result.append(coeff * C4[1] * (x2*y*z - 1/3*y3*z))
                    result.append(coeff * C4[2] * (x*y*z2 - 1/7*x*y))
                    result.append(coeff * C4[3] * (y*z3 - 3/7*y*z))
                    result.append(coeff * C4[4] * (z4 - 6/7*z2 + 3/35))
                    result.append(coeff * C4[5] * (x*z3 - 3/7*x*z))
                    result.append(coeff * C4[6] * (x2*z2 - 1/7*x2 - y2*z2 + (1/7)*y2))
                    result.append(coeff * C4[7] * ((1/3)*x3*z - x*y2*z))
                    result.append(coeff * C4[8] * ((1/6)*x4 - x2*y2 + (1/6)*y4))
                    if deg > 4:
                        x5, y5, z5 = x4 * x, y4 * y, z4 * z
                        result.append(coeff * C5[0] * ((1/2)*x4*y - x2*y3 + (1/10)*y5))
                        result.append(coeff * C5[1] * (x3*y*z - x*y3*z))
                        result.append(coeff * C5[2] * (x2*y*z2 - 1/9*x2*y - 1/3*y3*z2 + (1/27)*y3))
                        result.append(coeff * C5[3] * (x*y*z3 - 1/3*x*y*z))
                        result.append(coeff * C5[4] * (y*z4 - 2/3*y*z2 + (1/21)*y))
                        result.append(coeff * C5[5] * ((9/10)*z5 - z3 + (3/14)*z))
                        result.append(coeff * C5[6] * (x*z4 - 2/3*x*z2 + (1/21)*x))
                        result.append(coeff * C5[7] * (x2*z3 - 1/3*x2*z - y2*z3 + (1/3)*y2*z))
                        result.append(coeff * C5[8] * ((1/3)*x3*z2 - 1/27*x3 - x*y2*z2 + (1/9)*x*y2))
                        result.append(coeff * C5[9] * ((1/6)*x4*z - x2*y2*z + (1/6)*y4*z))
                        result.append(coeff * C5[10] * ((1/10)*x5 - x3*y2 + (1/2)*x*y4))
                        if deg > 5:
                            x6, y6, z6 = x5 * x, y5 * y, z5 * z
                            result.append(coeff * C6[0] * ((3/10)*x5*y - x3*y3 + (3/10)*x*y5))
                            result.append(coeff * C6[1] * ((1/2)*x4*y*z - x2*y3*z + (1/10)*y5*z))
                            result.append(coeff * C6[2] * (x3*y*z2 - 1/11*x3*y - x*y3*z2 + (1/11)*x*y3))
                            result.append(coeff * C6[3] * (x2*y*z3 - 3/11*x2*y*z - 1/3*y3*z3 + (1/11)*y3*z))
                            result.append(coeff * C6[4] * (x*y*z4 - 6/11*x*y*z2 + (1/33)*x*y))
                            result.append(coeff * C6[5] * (y*z5 - 10/11*y*z3 + (5/33)*y*z))
                            result.append(coeff * C6[6] * ((11/15)*z6 - z4 + (1/3)*z2 - 1/63))
                            result.append(coeff * C6[7] * (x*z5 - 10/11*x*z3 + (5/33)*x*z))
                            result.append(coeff * C6[8] * (x2*z4 - 6/11*x2*z2 + (1/33)*x2 - y2*z4 + (6/11)*y2*z2 - 1/33*y2))
                            result.append(coeff * C6[9] * ((1/3)*x3*z3 - 1/11*x3*z - x*y2*z3 + (3/11)*x*y2*z))
                            result.append(coeff * C6[10] * ((1/6)*x4*z2 - 1/66*x4 - x2*y2*z2 + (1/11)*x2*y2 + (1/6)*y4*z2 - 1/66*y4))
                            result.append(coeff * C6[11] * ((1/10)*x5*z - x3*y2*z + (1/2)*x*y4*z))
                            result.append(coeff * C6[12] * ((1/15)*x6 - x4*y2 + x2*y4 - 1/15*y6))
                            if deg > 6:
                                x7, y7, z7 = x6 * x, y6 * y, z6 * z
                                result.append(coeff * C7[0] * ((1/5)*x6*y - x4*y3 + (3/5)*x2*y5 - 1/35*y7))
                                result.append(coeff * C7[1] * ((3/10)*x5*y*z - x3*y3*z + (3/10)*x*y5*z))
                                result.append(coeff * C7[2] * ((1/2)*x4*y*z2 - 1/26*x4*y - x2*y3*z2 + (1/13)*x2*y3 + (1/10)*y5*z2 - 1/130*y5))
                                result.append(coeff * C7[3] * (x3*y*z3 - 3/13*x3*y*z - x*y3*z3 + (3/13)*x*y3*z))
                                result.append(coeff * C7[4] * (x2*y*z4 - 6/13*x2*y*z2 + (3/143)*x2*y - 1/3*y3*z4 + (2/13)*y3*z2 - 1/143*y3))
                                result.append(coeff * C7[5] * (x*y*z5 - 10/13*x*y*z3 + (15/143)*x*y*z))
                                result.append(coeff * C7[6] * ((13/15)*y*z6 - y*z4 + (3/11)*y*z2 - 1/99*y))
                                result.append(coeff * C7[7] * ((13/21)*z7 - z5 + (5/11)*z3 - 5/99*z))
                                result.append(coeff * C7[8] * ((13/15)*x*z6 - x*z4 + (3/11)*x*z2 - 1/99*x))
                                result.append(coeff * C7[9] * (x2*z5 - 10/13*x2*z3 + (15/143)*x2*z - y2*z5 + (10/13)*y2*z3 - 15/143*y2*z))
                                result.append(coeff * C7[10] * ((1/3)*x3*z4 - 2/13*x3*z2 + (1/143)*x3 - x*y2*z4 + (6/13)*x*y2*z2 - 3/143*x*y2))
                                result.append(coeff * C7[11] * ((1/6)*x4*z3 - 1/26*x4*z - x2*y2*z3 + (3/13)*x2*y2*z + (1/6)*y4*z3 - 1/26*y4*z))
                                result.append(coeff * C7[12] * ((1/10)*x5*z2 - 1/130*x5 - x3*y2*z2 + (1/13)*x3*y2 + (1/2)*x*y4*z2 - 1/26*x*y4))
                                result.append(coeff * C7[13] * ((1/15)*x6*z - x4*y2*z + x2*y4*z - 1/15*y6*z))
                                result.append(coeff * C7[14] * ((1/35)*x7 - 3/5*x5*y2 + x3*y4 - 1/5*x*y6))
                                if deg > 7:
                                    x8, y8, z8 = x7 * x, y7 * y, z7 * z
                                    result.append(coeff * C8[0] * ((1/7)*x7*y - x5*y3 + x3*y5 - 1/7*x*y7))
                                    result.append(coeff * C8[1] * ((1/5)*x6*y*z - x4*y3*z + (3/5)*x2*y5*z - 1/35*y7*z))
                                    result.append(coeff * C8[2] * ((3/10)*x5*y*z2 - 1/50*x5*y - x3*y3*z2 + (1/15)*x3*y3 + (3/10)*x*y5*z2 - 1/50*x*y5))
                                    result.append(coeff * C8[3] * ((1/2)*x4*y*z3 - 1/10*x4*y*z - x2*y3*z3 + (1/5)*x2*y3*z + (1/10)*y5*z3 - 1/50*y5*z))
                                    result.append(coeff * C8[4] * (x3*y*z4 - 2/5*x3*y*z2 + (1/65)*x3*y - x*y3*z4 + (2/5)*x*y3*z2 - 1/65*x*y3))
                                    result.append(coeff * C8[5] * (x2*y*z5 - 2/3*x2*y*z3 + (1/13)*x2*y*z - 1/3*y3*z5 + (2/9)*y3*z3 - 1/39*y3*z))
                                    result.append(coeff * C8[6] * (x*y*z6 - x*y*z4 + (3/13)*x*y*z2 - 1/143*x*y))
                                    result.append(coeff * C8[7] * ((5/7)*y*z7 - y*z5 + (5/13)*y*z3 - 5/143*y*z))
                                    result.append(coeff * C8[8] * ((15/28)*z8 - z6 + (15/26)*z4 - 15/143*z2 + 5/1716))
                                    result.append(coeff * C8[9] * ((5/7)*x*z7 - x*z5 + (5/13)*x*z3 - 5/143*x*z))
                                    result.append(coeff * C8[10] * (x2*z6 - x2*z4 + (3/13)*x2*z2 - 1/143*x2 - y2*z6 + y2*z4 - 3/13*y2*z2 + (1/143)*y2))
                                    result.append(coeff * C8[11] * ((1/3)*x3*z5 - 2/9*x3*z3 + (1/39)*x3*z - x*y2*z5 + (2/3)*x*y2*z3 - 1/13*x*y2*z))
                                    result.append(coeff * C8[12] * ((1/6)*x4*z4 - 1/15*x4*z2 + (1/390)*x4 - x2*y2*z4 + (2/5)*x2*y2*z2 - 1/65*x2*y2 + (1/6)*y4*z4 - 1/15*y4*z2 + (1/390)*y4))
                                    result.append(coeff * C8[13] * ((1/10)*x5*z3 - 1/50*x5*z - x3*y2*z3 + (1/5)*x3*y2*z + (1/2)*x*y4*z3 - 1/10*x*y4*z))
                                    result.append(coeff * C8[14] * ((1/15)*x6*z2 - 1/225*x6 - x4*y2*z2 + (1/15)*x4*y2 + x2*y4*z2 - 1/15*x2*y4 - 1/15*y6*z2 + (1/225)*y6))
                                    result.append(coeff * C8[15] * ((1/35)*x7*z - 3/5*x5*y2*z + x3*y4*z - 1/5*x*y6*z))
                                    result.append(coeff * C8[16] * ((1/70)*x8 - 2/5*x6*y2 + x4*y4 - 2/5*x2*y6 + (1/70)*y8))
                                    if deg > 8:
                                        x9, y9, z9 = x8 * x, y8 * y, z8 * z
                                        result.append(coeff * C9[0] * ((1/14)*x8*y - 2/3*x6*y3 + x4*y5 - 2/7*x2*y7 + (1/126)*y9))
                                        result.append(coeff * C9[1] * ((1/7)*x7*y*z - x5*y3*z + x3*y5*z - 1/7*x*y7*z))
                                        result.append(coeff * C9[2] * ((1/5)*x6*y*z2 - 1/85*x6*y - x4*y3*z2 + (1/17)*x4*y3 + (3/5)*x2*y5*z2 - 3/85*x2*y5 - 1/35*y7*z2 + (1/595)*y7))
                                        result.append(coeff * C9[3] * ((3/10)*x5*y*z3 - 9/170*x5*y*z - x3*y3*z3 + (3/17)*x3*y3*z + (3/10)*x*y5*z3 - 9/170*x*y5*z))
                                        result.append(coeff * C9[4] * ((1/2)*x4*y*z4 - 3/17*x4*y*z2 + (1/170)*x4*y - x2*y3*z4 + (6/17)*x2*y3*z2 - 1/85*x2*y3 + (1/10)*y5*z4 - 3/85*y5*z2 + (1/850)*y5))
                                        result.append(coeff * C9[5] * (x3*y*z5 - 10/17*x3*y*z3 + (1/17)*x3*y*z - x*y3*z5 + (10/17)*x*y3*z3 - 1/17*x*y3*z))
                                        result.append(coeff * C9[6] * (x2*y*z6 - 15/17*x2*y*z4 + (3/17)*x2*y*z2 - 1/221*x2*y - 1/3*y3*z6 + (5/17)*y3*z4 - 1/17*y3*z2 + (1/663)*y3))
                                        result.append(coeff * C9[7] * ((17/21)*x*y*z7 - x*y*z5 + (1/3)*x*y*z3 - 1/39*x*y*z))
                                        result.append(coeff * C9[8] * ((17/28)*y*z8 - y*z6 + (1/2)*y*z4 - 1/13*y*z2 + (1/572)*y))
                                        result.append(coeff * C9[9] * ((17/36)*z9 - z7 + (7/10)*z5 - 7/39*z3 + (7/572)*z))
                                        result.append(coeff * C9[10] * ((17/28)*x*z8 - x*z6 + (1/2)*x*z4 - 1/13*x*z2 + (1/572)*x))
                                        result.append(coeff * C9[11] * ((17/21)*x2*z7 - x2*z5 + (1/3)*x2*z3 - 1/39*x2*z - 17/21*y2*z7 + y2*z5 - 1/3*y2*z3 + (1/39)*y2*z))
                                        result.append(coeff * C9[12] * ((1/3)*x3*z6 - 5/17*x3*z4 + (1/17)*x3*z2 - 1/663*x3 - x*y2*z6 + (15/17)*x*y2*z4 - 3/17*x*y2*z2 + (1/221)*x*y2))
                                        result.append(coeff * C9[13] * ((1/6)*x4*z5 - 5/51*x4*z3 + (1/102)*x4*z - x2*y2*z5 + (10/17)*x2*y2*z3 - 1/17*x2*y2*z + (1/6)*y4*z5 - 5/51*y4*z3 + (1/102)*y4*z))
                                        result.append(coeff * C9[14] * ((1/10)*x5*z4 - 3/85*x5*z2 + (1/850)*x5 - x3*y2*z4 + (6/17)*x3*y2*z2 - 1/85*x3*y2 + (1/2)*x*y4*z4 - 3/17*x*y4*z2 + (1/170)*x*y4))
                                        result.append(coeff * C9[15] * ((1/15)*x6*z3 - 1/85*x6*z - x4*y2*z3 + (3/17)*x4*y2*z + x2*y4*z3 - 3/17*x2*y4*z - 1/15*y6*z3 + (1/85)*y6*z))
                                        result.append(coeff * C9[16] * ((1/35)*x7*z2 - 1/595*x7 - 3/5*x5*y2*z2 + (3/85)*x5*y2 + x3*y4*z2 - 1/17*x3*y4 - 1/5*x*y6*z2 + (1/85)*x*y6))
                                        result.append(coeff * C9[17] * ((1/70)*x8*z - 2/5*x6*y2*z + x4*y4*z - 2/5*x2*y6*z + (1/70)*y8*z))
                                        result.append(coeff * C9[18] * ((1/126)*x9 - 2/7*x7*y2 + x5*y4 - 2/3*x3*y6 + (1/14)*x*y8))
    return torch.stack(result, dim=-1)

def eval_sh(deg, sh, dirs):
    assert deg <= 9 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    dirs = torch.where(dirs[..., 2:3] > 0, dirs, -dirs)

    result = C0[0] * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result + C1[0] * (y) * sh[..., 1] + 
                C1[1] * (z) * sh[..., 2] + 
                C1[2] * (x) * sh[..., 3])
        if deg > 1:
            x2, y2, z2 = x * x, y * y, z * z
            result = (result + C2[0] * (x*y) * sh[..., 4] + 
                    C2[1] * (y*z) * sh[..., 5] + 
                    C2[2] * (z2 - 1/3) * sh[..., 6] + 
                    C2[3] * (x*z) * sh[..., 7] + 
                    C2[4] * (x2 - y2) * sh[..., 8])
            if deg > 2:
                x3, y3, z3 = x2 * x, y2 * y, z2 * z
                result = (result + C3[0] * (x2*y - 1/3*y3) * sh[..., 9] + 
                        C3[1] * (x*y*z) * sh[..., 10] + 
                        C3[2] * (y*z2 - 1/5*y) * sh[..., 11] + 
                        C3[3] * (z3 - 3/5*z) * sh[..., 12] + 
                        C3[4] * (x*z2 - 1/5*x) * sh[..., 13] + 
                        C3[5] * (x2*z - y2*z) * sh[..., 14] + 
                        C3[6] * ((1/3)*x3 - x*y2) * sh[..., 15])
                if deg > 3:
                    x4, y4, z4 = x3 * x, y3 * y, z3 * z
                    result = (result + C4[0] * (x3*y - x*y3) * sh[..., 16] + 
                            C4[1] * (x2*y*z - 1/3*y3*z) * sh[..., 17] + 
                            C4[2] * (x*y*z2 - 1/7*x*y) * sh[..., 18] + 
                            C4[3] * (y*z3 - 3/7*y*z) * sh[..., 19] + 
                            C4[4] * (z4 - 6/7*z2 + 3/35) * sh[..., 20] + 
                            C4[5] * (x*z3 - 3/7*x*z) * sh[..., 21] + 
                            C4[6] * (x2*z2 - 1/7*x2 - y2*z2 + (1/7)*y2) * sh[..., 22] + 
                            C4[7] * ((1/3)*x3*z - x*y2*z) * sh[..., 23] + 
                            C4[8] * ((1/6)*x4 - x2*y2 + (1/6)*y4) * sh[..., 24])
                    if deg > 4:
                        x5, y5, z5 = x4 * x, y4 * y, z4 * z
                        result = (result + C5[0] * ((1/2)*x4*y - x2*y3 + (1/10)*y5) * sh[..., 25] + 
                                C5[1] * (x3*y*z - x*y3*z) * sh[..., 26] + 
                                C5[2] * (x2*y*z2 - 1/9*x2*y - 1/3*y3*z2 + (1/27)*y3) * sh[..., 27] + 
                                C5[3] * (x*y*z3 - 1/3*x*y*z) * sh[..., 28] + 
                                C5[4] * (y*z4 - 2/3*y*z2 + (1/21)*y) * sh[..., 29] + 
                                C5[5] * ((9/10)*z5 - z3 + (3/14)*z) * sh[..., 30] + 
                                C5[6] * (x*z4 - 2/3*x*z2 + (1/21)*x) * sh[..., 31] + 
                                C5[7] * (x2*z3 - 1/3*x2*z - y2*z3 + (1/3)*y2*z) * sh[..., 32] + 
                                C5[8] * ((1/3)*x3*z2 - 1/27*x3 - x*y2*z2 + (1/9)*x*y2) * sh[..., 33] + 
                                C5[9] * ((1/6)*x4*z - x2*y2*z + (1/6)*y4*z) * sh[..., 34] + 
                                C5[10] * ((1/10)*x5 - x3*y2 + (1/2)*x*y4) * sh[..., 35])
                        if deg > 5:
                            x6, y6, z6 = x5 * x, y5 * y, z5 * z
                            result = (result + C6[0] * ((3/10)*x5*y - x3*y3 + (3/10)*x*y5) * sh[..., 36] + 
                                    C6[1] * ((1/2)*x4*y*z - x2*y3*z + (1/10)*y5*z) * sh[..., 37] + 
                                    C6[2] * (x3*y*z2 - 1/11*x3*y - x*y3*z2 + (1/11)*x*y3) * sh[..., 38] + 
                                    C6[3] * (x2*y*z3 - 3/11*x2*y*z - 1/3*y3*z3 + (1/11)*y3*z) * sh[..., 39] + 
                                    C6[4] * (x*y*z4 - 6/11*x*y*z2 + (1/33)*x*y) * sh[..., 40] + 
                                    C6[5] * (y*z5 - 10/11*y*z3 + (5/33)*y*z) * sh[..., 41] + 
                                    C6[6] * ((11/15)*z6 - z4 + (1/3)*z2 - 1/63) * sh[..., 42] + 
                                    C6[7] * (x*z5 - 10/11*x*z3 + (5/33)*x*z) * sh[..., 43] + 
                                    C6[8] * (x2*z4 - 6/11*x2*z2 + (1/33)*x2 - y2*z4 + (6/11)*y2*z2 - 1/33*y2) * sh[..., 44] + 
                                    C6[9] * ((1/3)*x3*z3 - 1/11*x3*z - x*y2*z3 + (3/11)*x*y2*z) * sh[..., 45] + 
                                    C6[10] * ((1/6)*x4*z2 - 1/66*x4 - x2*y2*z2 + (1/11)*x2*y2 + (1/6)*y4*z2 - 1/66*y4) * sh[..., 46] + 
                                    C6[11] * ((1/10)*x5*z - x3*y2*z + (1/2)*x*y4*z) * sh[..., 47] + 
                                    C6[12] * ((1/15)*x6 - x4*y2 + x2*y4 - 1/15*y6) * sh[..., 48])
                            if deg > 6:
                                x7, y7, z7 = x6 * x, y6 * y, z6 * z
                                result = (result + C7[0] * ((1/5)*x6*y - x4*y3 + (3/5)*x2*y5 - 1/35*y7) * sh[..., 49] + 
                                        C7[1] * ((3/10)*x5*y*z - x3*y3*z + (3/10)*x*y5*z) * sh[..., 50] + 
                                        C7[2] * ((1/2)*x4*y*z2 - 1/26*x4*y - x2*y3*z2 + (1/13)*x2*y3 + (1/10)*y5*z2 - 1/130*y5) * sh[..., 51] + 
                                        C7[3] * (x3*y*z3 - 3/13*x3*y*z - x*y3*z3 + (3/13)*x*y3*z) * sh[..., 52] + 
                                        C7[4] * (x2*y*z4 - 6/13*x2*y*z2 + (3/143)*x2*y - 1/3*y3*z4 + (2/13)*y3*z2 - 1/143*y3) * sh[..., 53] + 
                                        C7[5] * (x*y*z5 - 10/13*x*y*z3 + (15/143)*x*y*z) * sh[..., 54] + 
                                        C7[6] * ((13/15)*y*z6 - y*z4 + (3/11)*y*z2 - 1/99*y) * sh[..., 55] + 
                                        C7[7] * ((13/21)*z7 - z5 + (5/11)*z3 - 5/99*z) * sh[..., 56] + 
                                        C7[8] * ((13/15)*x*z6 - x*z4 + (3/11)*x*z2 - 1/99*x) * sh[..., 57] + 
                                        C7[9] * (x2*z5 - 10/13*x2*z3 + (15/143)*x2*z - y2*z5 + (10/13)*y2*z3 - 15/143*y2*z) * sh[..., 58] + 
                                        C7[10] * ((1/3)*x3*z4 - 2/13*x3*z2 + (1/143)*x3 - x*y2*z4 + (6/13)*x*y2*z2 - 3/143*x*y2) * sh[..., 59] + 
                                        C7[11] * ((1/6)*x4*z3 - 1/26*x4*z - x2*y2*z3 + (3/13)*x2*y2*z + (1/6)*y4*z3 - 1/26*y4*z) * sh[..., 60] + 
                                        C7[12] * ((1/10)*x5*z2 - 1/130*x5 - x3*y2*z2 + (1/13)*x3*y2 + (1/2)*x*y4*z2 - 1/26*x*y4) * sh[..., 61] + 
                                        C7[13] * ((1/15)*x6*z - x4*y2*z + x2*y4*z - 1/15*y6*z) * sh[..., 62] + 
                                        C7[14] * ((1/35)*x7 - 3/5*x5*y2 + x3*y4 - 1/5*x*y6) * sh[..., 63])
                                if deg > 7:
                                    x8, y8, z8 = x7 * x, y7 * y, z7 * z
                                    result = (result + C8[0] * ((1/7)*x7*y - x5*y3 + x3*y5 - 1/7*x*y7) * sh[..., 64] + 
                                            C8[1] * ((1/5)*x6*y*z - x4*y3*z + (3/5)*x2*y5*z - 1/35*y7*z) * sh[..., 65] + 
                                            C8[2] * ((3/10)*x5*y*z2 - 1/50*x5*y - x3*y3*z2 + (1/15)*x3*y3 + (3/10)*x*y5*z2 - 1/50*x*y5) * sh[..., 66] + 
                                            C8[3] * ((1/2)*x4*y*z3 - 1/10*x4*y*z - x2*y3*z3 + (1/5)*x2*y3*z + (1/10)*y5*z3 - 1/50*y5*z) * sh[..., 67] + 
                                            C8[4] * (x3*y*z4 - 2/5*x3*y*z2 + (1/65)*x3*y - x*y3*z4 + (2/5)*x*y3*z2 - 1/65*x*y3) * sh[..., 68] + 
                                            C8[5] * (x2*y*z5 - 2/3*x2*y*z3 + (1/13)*x2*y*z - 1/3*y3*z5 + (2/9)*y3*z3 - 1/39*y3*z) * sh[..., 69] + 
                                            C8[6] * (x*y*z6 - x*y*z4 + (3/13)*x*y*z2 - 1/143*x*y) * sh[..., 70] + 
                                            C8[7] * ((5/7)*y*z7 - y*z5 + (5/13)*y*z3 - 5/143*y*z) * sh[..., 71] + 
                                            C8[8] * ((15/28)*z8 - z6 + (15/26)*z4 - 15/143*z2 + 5/1716) * sh[..., 72] + 
                                            C8[9] * ((5/7)*x*z7 - x*z5 + (5/13)*x*z3 - 5/143*x*z) * sh[..., 73] + 
                                            C8[10] * (x2*z6 - x2*z4 + (3/13)*x2*z2 - 1/143*x2 - y2*z6 + y2*z4 - 3/13*y2*z2 + (1/143)*y2) * sh[..., 74] + 
                                            C8[11] * ((1/3)*x3*z5 - 2/9*x3*z3 + (1/39)*x3*z - x*y2*z5 + (2/3)*x*y2*z3 - 1/13*x*y2*z) * sh[..., 75] + 
                                            C8[12] * ((1/6)*x4*z4 - 1/15*x4*z2 + (1/390)*x4 - x2*y2*z4 + (2/5)*x2*y2*z2 - 1/65*x2*y2 + (1/6)*y4*z4 - 1/15*y4*z2 + (1/390)*y4) * sh[..., 76] + 
                                            C8[13] * ((1/10)*x5*z3 - 1/50*x5*z - x3*y2*z3 + (1/5)*x3*y2*z + (1/2)*x*y4*z3 - 1/10*x*y4*z) * sh[..., 77] + 
                                            C8[14] * ((1/15)*x6*z2 - 1/225*x6 - x4*y2*z2 + (1/15)*x4*y2 + x2*y4*z2 - 1/15*x2*y4 - 1/15*y6*z2 + (1/225)*y6) * sh[..., 78] + 
                                            C8[15] * ((1/35)*x7*z - 3/5*x5*y2*z + x3*y4*z - 1/5*x*y6*z) * sh[..., 79] + 
                                            C8[16] * ((1/70)*x8 - 2/5*x6*y2 + x4*y4 - 2/5*x2*y6 + (1/70)*y8) * sh[..., 80])
                                    if deg > 8:
                                        x9, y9, z9 = x8 * x, y8 * y, z8 * z
                                        result = (result + C9[0] * ((1/14)*x8*y - 2/3*x6*y3 + x4*y5 - 2/7*x2*y7 + (1/126)*y9) * sh[..., 81] + 
                                                C9[1] * ((1/7)*x7*y*z - x5*y3*z + x3*y5*z - 1/7*x*y7*z) * sh[..., 82] + 
                                                C9[2] * ((1/5)*x6*y*z2 - 1/85*x6*y - x4*y3*z2 + (1/17)*x4*y3 + (3/5)*x2*y5*z2 - 3/85*x2*y5 - 1/35*y7*z2 + (1/595)*y7) * sh[..., 83] + 
                                                C9[3] * ((3/10)*x5*y*z3 - 9/170*x5*y*z - x3*y3*z3 + (3/17)*x3*y3*z + (3/10)*x*y5*z3 - 9/170*x*y5*z) * sh[..., 84] + 
                                                C9[4] * ((1/2)*x4*y*z4 - 3/17*x4*y*z2 + (1/170)*x4*y - x2*y3*z4 + (6/17)*x2*y3*z2 - 1/85*x2*y3 + (1/10)*y5*z4 - 3/85*y5*z2 + (1/850)*y5) * sh[..., 85] + 
                                                C9[5] * (x3*y*z5 - 10/17*x3*y*z3 + (1/17)*x3*y*z - x*y3*z5 + (10/17)*x*y3*z3 - 1/17*x*y3*z) * sh[..., 86] + 
                                                C9[6] * (x2*y*z6 - 15/17*x2*y*z4 + (3/17)*x2*y*z2 - 1/221*x2*y - 1/3*y3*z6 + (5/17)*y3*z4 - 1/17*y3*z2 + (1/663)*y3) * sh[..., 87] + 
                                                C9[7] * ((17/21)*x*y*z7 - x*y*z5 + (1/3)*x*y*z3 - 1/39*x*y*z) * sh[..., 88] + 
                                                C9[8] * ((17/28)*y*z8 - y*z6 + (1/2)*y*z4 - 1/13*y*z2 + (1/572)*y) * sh[..., 89] + 
                                                C9[9] * ((17/36)*z9 - z7 + (7/10)*z5 - 7/39*z3 + (7/572)*z) * sh[..., 90] + 
                                                C9[10] * ((17/28)*x*z8 - x*z6 + (1/2)*x*z4 - 1/13*x*z2 + (1/572)*x) * sh[..., 91] + 
                                                C9[11] * ((17/21)*x2*z7 - x2*z5 + (1/3)*x2*z3 - 1/39*x2*z - 17/21*y2*z7 + y2*z5 - 1/3*y2*z3 + (1/39)*y2*z) * sh[..., 92] + 
                                                C9[12] * ((1/3)*x3*z6 - 5/17*x3*z4 + (1/17)*x3*z2 - 1/663*x3 - x*y2*z6 + (15/17)*x*y2*z4 - 3/17*x*y2*z2 + (1/221)*x*y2) * sh[..., 93] + 
                                                C9[13] * ((1/6)*x4*z5 - 5/51*x4*z3 + (1/102)*x4*z - x2*y2*z5 + (10/17)*x2*y2*z3 - 1/17*x2*y2*z + (1/6)*y4*z5 - 5/51*y4*z3 + (1/102)*y4*z) * sh[..., 94] + 
                                                C9[14] * ((1/10)*x5*z4 - 3/85*x5*z2 + (1/850)*x5 - x3*y2*z4 + (6/17)*x3*y2*z2 - 1/85*x3*y2 + (1/2)*x*y4*z4 - 3/17*x*y4*z2 + (1/170)*x*y4) * sh[..., 95] + 
                                                C9[15] * ((1/15)*x6*z3 - 1/85*x6*z - x4*y2*z3 + (3/17)*x4*y2*z + x2*y4*z3 - 3/17*x2*y4*z - 1/15*y6*z3 + (1/85)*y6*z) * sh[..., 96] + 
                                                C9[16] * ((1/35)*x7*z2 - 1/595*x7 - 3/5*x5*y2*z2 + (3/85)*x5*y2 + x3*y4*z2 - 1/17*x3*y4 - 1/5*x*y6*z2 + (1/85)*x*y6) * sh[..., 97] + 
                                                C9[17] * ((1/70)*x8*z - 2/5*x6*y2*z + x4*y4*z - 2/5*x2*y6*z + (1/70)*y8*z) * sh[..., 98] + 
                                                C9[18] * ((1/126)*x9 - 2/7*x7*y2 + x5*y4 - 2/3*x3*y6 + (1/14)*x*y8) * sh[..., 99])
    return torch.clamp_min(result, 0.)

def RGB2SH(rgb):
    return rgb / C0[0]

def SH2RGB(sh):
    return sh * C0[0]