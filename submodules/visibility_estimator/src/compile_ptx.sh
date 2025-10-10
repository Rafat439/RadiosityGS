#! /bin/bash

nvcc -keep -o optix_dev.ptx optix_dev.cu -I ../third_party/glm -I ../third_party/optix/include -lcurand -loptix -lcublas -lcublas_device -lcudadevrt --use_fast_math -maxrregcount 128 -O3 -ptx -DNDEBUG
xxd -i optix_dev.ptx >optix_dev_ptx.h