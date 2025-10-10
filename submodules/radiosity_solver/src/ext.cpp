#include <torch/extension.h>
#include "next_event.h"
#include "solver.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("next_event_estimator", &NextEventEstimatorCUDA);
    m.def("in_cluster_next_event_estimator", &InClusterNextEventEstimatorCUDA);
    m.def("optix_build_bvh", &OptixBuildBVH);
    m.def("optix_release_bvh", &OptixReleaseBVH);
    m.def("fused_backward", &FusedBackward);
}