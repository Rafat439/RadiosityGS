from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="visibility_estimator", 
    ext_modules=[
        CUDAExtension(
            name="_visibility_estimator", 
            sources=[
            "src/optix_estimator.cu", 
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