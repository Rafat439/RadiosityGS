#include <torch/extension.h>
#include "optix_estimator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optix_build_bvh", &OptixBuildBVH);
    m.def("optix_release_bvh", &OptixReleaseBVH);
    m.def("optix_estimate_visibility_forward", &OptixVisibilityEstimatorForwardCUDA);
    m.def("optix_estimate_visibility_backward", &OptixVisibilityEstimatorBackwardCUDA);
}