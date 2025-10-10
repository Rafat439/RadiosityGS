from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="radiosity_solver_aux", 
    ext_modules=[
        CUDAExtension(
            name="_radiosity_solver_aux", 
            sources=[
            "src/next_event.cu",
            "src/solver.cu", 
            "src/ext.cpp"],
            libraries=["cuda"], 
            library_path=["/usr/local/cuda/lib64/stubs"], 
            extra_compile_args={"nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"), 
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/optix/include"),
                "-lcurand", 
                "-loptix", 
                '-lcublas', 
                '-lcublas_device', 
                '-lcudadevrt', 
                "--use_fast_math", 
                "-O3"]}, 
            py_limited_api=True
        )
    ], 
    cmdclass={
        'build_ext': BuildExtension
    }
)