#ifdef BACKEND_CUDNN
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"