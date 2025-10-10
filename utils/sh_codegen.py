import math
import sympy
from sympy.codegen import ast
from sympy.codegen import rewriting
from sympy.functions.special.polynomials import legendre

sympy.init_printing(use_unicode=True)

x, y, z = sympy.symbols("x y z", real=True)
i = sympy.I
cnt = 0
left_par = "{"
right_par = "}"
constant_definitions = ""
forward_code_snippets = ""
solver_forward_code_snippets = ""
backward_code_snippts = ""
solver_backward_code_snippts = ""

python_constant_definitions = ""
python_code_snippets = ""
python_solver_code_snippets = ""

def C(l, m):
    assert m >= 0
    return math.sqrt(((2 * l + 1) / (4 * math.pi)) * (math.factorial(l - m) / math.factorial(l + m)))

def P(l, m):
    assert m >= 0
    P_l = legendre(l, z)
    # print(f'At {l}, we have {P_l}')
    for _ in range(m):
        P_l = sympy.diff(P_l, z)
        # print(f' {_}: {P_l}')
    return ((-1) ** m) * P_l

def Y(l, m):
    assert m >= 0
    return P(l, m) * ((x + i * y) ** m)

def convert_to_aggr(L, raw_expr_s, raw_expr_dX_s, raw_expr_dY_s, raw_expr_dZ_s, python_raw_expr_s, tabs):
    def process_exp(e):
        for l in range(L + 1):
            e = e.replace(f"powf(x, {l})", f"x{l}")
            e = e.replace(f"powf(y, {l})", f"y{l}")
            e = e.replace(f"powf(z, {l})", f"z{l}")
            e = e.replace(f"x**{l}", f"x{l}")
            e = e.replace(f"y**{l}", f"y{l}")
            e = e.replace(f"z**{l}", f"z{l}")
        return e
    global cnt
    sf_expr_s = []
    f_expr_s = []
    b_expr_s = []
    sp_expr_s = []
    p_expr_s = []
    dX_s = []
    dY_s = []
    dZ_s = []
    for i, (e, ex, ey, ez, pe) in enumerate(zip(raw_expr_s, raw_expr_dX_s, raw_expr_dY_s, raw_expr_dZ_s, python_raw_expr_s)):
        e = process_exp(e)
        ex = process_exp(ex)
        ey = process_exp(ey)
        ez = process_exp(ez)
        pe = process_exp(pe)
        # print(e, ex, ey, ez)
        if e == '0':
            pass
        elif e == '1':
            f_expr_s.append(f"SH_C{L}[{i}] * sh[{cnt}]")
            p_expr_s.append(f"C{L}[{i}] * sh[..., {cnt}]")
            sf_expr_s.append(f"result[{cnt}] = coeff * SH_C{L}[{i}] * brdf_coeffs[{cnt}];\n")
            sp_expr_s.append(f"result.append(coeff * C{L}[{i}])\n")
            b_expr_s.append(f"dL_dsh[{cnt}] = SH_C{L}[{i}] * dL_dRGB;\n")
        else:
            f_expr_s.append(f"SH_C{L}[{i}] * (" + e + f") * sh[{cnt}]")
            p_expr_s.append(f"C{L}[{i}] * (" + pe + f") * sh[..., {cnt}]")
            sf_expr_s.append(f"result[{cnt}] = coeff * SH_C{L}[{i}] * (" + e + f") * brdf_coeffs[{cnt}];\n")
            sp_expr_s.append(f"result.append(coeff * C{L}[{i}] * (" + pe + f"))\n")
            b_expr_s.append(f"dL_dsh[{cnt}] = SH_C{L}[{i}] * (" + e + ") * dL_dRGB;\n")
        
        if ex == '0':
            pass
        elif ex == '1':
            dX_s.append(f"SH_C{L}[{i}] * sh[{cnt}]")
        else:
            dX_s.append(f"SH_C{L}[{i}] * (" + ex + f") * sh[{cnt}]")
        
        if ey == '0':
            pass
        elif ey == '1':
            dY_s.append(f"SH_C{L}[{i}] * sh[{cnt}]")
        else:
            dY_s.append(f"SH_C{L}[{i}] * (" + ey + f") * sh[{cnt}]")
        
        if ez == '0':
            pass
        elif ez == '1':
            dZ_s.append(f"SH_C{L}[{i}] * sh[{cnt}]")
        else:
            dZ_s.append(f"SH_C{L}[{i}] * (" + ez + f") * sh[{cnt}]")
        
        cnt += 1
    return \
        f"{tabs}result += " + f" + \n{tabs}          ".join(f_expr_s) + ";\n", \
        tabs + f"{tabs}".join(sf_expr_s), \
        tabs + f"{tabs}".join(b_expr_s), \
        ((f"{tabs}dRGBdx += " + " + ".join(dX_s) + f";\n") if len(dX_s) > 0 else "") + \
        ((f"{tabs}dRGBdy += " + " + ".join(dY_s) + f";\n") if len(dY_s) > 0 else "") + \
        ((f"{tabs}dRGBdz += " + " + ".join(dZ_s) + f";\n") if len(dZ_s) > 0 else "") + "\n", \
        f"{tabs}result = (result + " + f" + \n{tabs}          ".join(p_expr_s) + ")\n", \
        tabs + f"{tabs}".join(sp_expr_s)

def generate_code_for_L(L, max_L, tabs):
    global constant_definitions
    global forward_code_snippets
    global solver_forward_code_snippets
    global backward_code_snippts
    global solver_backward_code_snippts
    global python_constant_definitions
    global python_code_snippets
    global python_solver_code_snippets
    
    if L > max_L:
        return
    
    C_R = []
    Y_R = []
    dY_dX_R = []
    dY_dY_R = []
    dY_dZ_R = []
    for m in range(-L, L+1):
        if m == 0:
            _c = C(L, m)
            _y = Y(L, m)
        elif m > 0:
            _c = math.sqrt(2) * C(L, m)
            _y = sympy.nsimplify(sympy.expand(((-1) ** m) * sympy.re(Y(L, m))))
        elif m < 0:
            _c = math.sqrt(2) * C(L, -m)
            _y = sympy.nsimplify(sympy.expand(((-1) ** m) * sympy.im(Y(L, -m))))
        _y = sympy.Poly(sympy.factor(_y), x, y, z)
        hcf = max( [abs(coeff) for coeff in _y.coeffs()] )
        _c = _c * hcf
        _y = (_y / hcf).as_expr()
        C_R.append(str(_c) + "F")
        Y_R.append(_y)
        dY_dX_R.append(sympy.expand(sympy.diff(_y, x)))
        dY_dY_R.append(sympy.expand(sympy.diff(_y, y)))
        dY_dZ_R.append(sympy.expand(sympy.diff(_y, z)))
    constant_definitions += f"__device__ const float SH_C{L}[] = {{{', '.join(C_R)}}};\n"
    python_constant_definitions += f"C{L} = [{', '.join(C_R)}]\n".replace("F", "")
    
    raw_expr_s = [sympy.cxxcode(rewriting.optimize(_y, rewriting.optims_c99), type_aliases={ast.real: ast.float32}).replace("std::", "") for _y in Y_R]
    python_raw_expr_s = [sympy.pycode(_y) for _y in Y_R]
    raw_expr_dX_s = [sympy.cxxcode(rewriting.optimize(_y, rewriting.optims_c99), type_aliases={ast.real: ast.float32}).replace("std::", "") for _y in dY_dX_R]
    raw_expr_dY_s = [sympy.cxxcode(rewriting.optimize(_y, rewriting.optims_c99), type_aliases={ast.real: ast.float32}).replace("std::", "") for _y in dY_dY_R]
    raw_expr_dZ_s = [sympy.cxxcode(rewriting.optimize(_y, rewriting.optims_c99), type_aliases={ast.real: ast.float32}).replace("std::", "") for _y in dY_dZ_R]
    if L > 0:
        forward_code_snippets += f"{tabs}if (deg > {L - 1})\n{tabs}{left_par}\n"
        solver_forward_code_snippets += f"{tabs}if (deg > {L - 1})\n{tabs}{left_par}\n"
        backward_code_snippts += f"{tabs}if (deg > {L - 1})\n{tabs}{left_par}\n"
        solver_backward_code_snippts += f"{tabs}if (deg > {L - 1})\n{tabs}{left_par}\n"
        python_code_snippets += f"{tabs}if deg > {L - 1}:\n"
        python_solver_code_snippets += f"{tabs}if deg > {L - 1}:\n"
        tabs += "\t"
        if L == 2:
            forward_code_snippets += f"{tabs}float x{L} = x * x;\n{tabs}float y{L} = y * y;\n{tabs}float z{L} = z * z;\n"
            solver_forward_code_snippets += f"{tabs}float x{L} = x * x;\n{tabs}float y{L} = y * y;\n{tabs}float z{L} = z * z;\n"
            backward_code_snippts += f"{tabs}float x{L} = x * x;\n{tabs}float y{L} = y * y;\n{tabs}float z{L} = z * z;\n"
            solver_backward_code_snippts += f"{tabs}float x{L} = x * x;\n{tabs}float y{L} = y * y;\n{tabs}float z{L} = z * z;\n"
            python_code_snippets += f"{tabs}x2, y2, z2 = x * x, y * y, z * z\n"
            python_solver_code_snippets += f"{tabs}x2, y2, z2 = x * x, y * y, z * z\n"
        elif L > 2:
            forward_code_snippets += f"{tabs}float x{L} = x{L - 1} * x;\n{tabs}float y{L} = y{L - 1} * y;\n{tabs}float z{L} = z{L - 1} * z;\n"
            solver_forward_code_snippets += f"{tabs}float x{L} = x{L - 1} * x;\n{tabs}float y{L} = y{L - 1} * y;\n{tabs}float z{L} = z{L - 1} * z;\n"
            backward_code_snippts += f"{tabs}float x{L} = x{L - 1} * x;\n{tabs}float y{L} = y{L - 1} * y;\n{tabs}float z{L} = z{L - 1} * z;\n"
            solver_backward_code_snippts += f"{tabs}float x{L} = x{L - 1} * x;\n{tabs}float y{L} = y{L - 1} * y;\n{tabs}float z{L} = z{L - 1} * z;\n"
            python_code_snippets += f"{tabs}x{L}, y{L}, z{L} = x{L - 1} * x, y{L - 1} * y, z{L - 1} * z\n"
            python_solver_code_snippets += f"{tabs}x{L}, y{L}, z{L} = x{L - 1} * x, y{L - 1} * y, z{L - 1} * z\n"
    f_c, sf_c, b_c, sb_c, p_c, sp_c = convert_to_aggr(L, raw_expr_s, raw_expr_dX_s, raw_expr_dY_s, raw_expr_dZ_s, python_raw_expr_s, tabs)
    forward_code_snippets += f_c
    solver_forward_code_snippets += sf_c
    backward_code_snippts += b_c + "\n" + sb_c
    solver_backward_code_snippts += sb_c
    python_code_snippets += p_c
    python_solver_code_snippets += sp_c
    generate_code_for_L(L + 1, max_L, tabs)
    if L > 0:
        forward_code_snippets += f"{tabs[:-1]}{right_par}\n"
        solver_forward_code_snippets += f"{tabs[:-1]}{right_par}\n"
        backward_code_snippts += f"{tabs[:-1]}{right_par}\n"
        solver_backward_code_snippts += f"{tabs[:-1]}{right_par}\n"

if __name__ == "__main__":
    generate_code_for_L(0, 9, "")
    with open("_sh_constants.cu", "w") as f:
        f.write(constant_definitions)
    with open("_sh_forward.cu", "w") as f:
        f.write(forward_code_snippets)
    with open("_sh_solver_forward.cu", "w") as f:
        f.write(solver_forward_code_snippets)
    with open("_sh_backward.cu", "w") as f:
        f.write(backward_code_snippts)
    with open("_sh_solver_backward.cu", "w") as f:
        f.write(solver_backward_code_snippts)
    with open("_sh_python_constants.py", "w") as f:
        f.write(python_constant_definitions)
    with open("_sh_python_code.py", "w") as f:
        f.write(python_code_snippets)
    with open("_sh_python_solver_code.py", "w") as f:
        f.write(python_solver_code_snippets)