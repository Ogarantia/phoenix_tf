#ifdef BACKEND_CUDNN
#define EIGEN_USE_GPU
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"       // disabling tensorflow warnings

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#pragma GCC diagnostic pop