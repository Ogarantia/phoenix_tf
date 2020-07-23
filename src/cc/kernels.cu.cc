#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "upstride.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void UpstrideInputCudaKernel(const int size,
                           const T* input_1,
                           const T* input_2,
                           const T* input_3,
                           const T* input_4,
                           T* input_processed_1,
                           T* input_processed_2,
                           T* input_processed_3,
                           T* input_processed_4,
                           T* input_processed_5,
                           T* input_processed_6,
                           T* input_processed_7,
                           T* input_processed_8) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        input_processed_1[i] = input_4[i] + input_2[i];
        input_processed_2[i] = input_1[i] - input_3[i];
        input_processed_3[i] = input_1[i] + input_3[i];
        input_processed_4[i] = input_4[i] - input_2[i];
        input_processed_5[i] = input_4[i] - input_3[i];
        input_processed_6[i] = input_2[i] + input_1[i];
        input_processed_7[i] = input_1[i] - input_2[i];
        input_processed_8[i] = input_4[i] + input_3[i];
    }
}

template <typename T>
__global__ void UpstrideKernelCudaKernel(const int size,
                                      const T* kernel_1,
                                      const T* kernel_2,
                                      const T* kernel_3,
                                      const T* kernel_4,
                                      T* kernel_processed_1,
                                      T* kernel_processed_2,
                                      T* kernel_processed_3,
                                      T* kernel_processed_4,
                                      T* kernel_processed_5,
                                      T* kernel_processed_6,
                                      T* kernel_processed_7,
                                      T* kernel_processed_8) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        kernel_processed_1[i] = kernel_2[i] + kernel_3[i];
        kernel_processed_2[i] = kernel_1[i] + kernel_4[i];
        kernel_processed_3[i] = kernel_1[i] - kernel_4[i];
        kernel_processed_4[i] = kernel_2[i] - kernel_3[i];
        kernel_processed_5[i] = kernel_3[i] - kernel_4[i];
        kernel_processed_6[i] = kernel_2[i] + kernel_1[i];
        kernel_processed_7[i] = kernel_3[i] + kernel_4[i];
        kernel_processed_8[i] = kernel_1[i] - kernel_2[i];
    }
}

template <typename T>
__global__ void UpstrideOutputCudaKernel(const int size,
                                      const T* output_1,
                                      const T* output_2,
                                      const T* output_3,
                                      const T* output_4,
                                      const T* output_5,
                                      const T* output_6,
                                      const T* output_7,
                                      const T* output_8,
                                      T* output_processed_1,
                                      T* output_processed_2,
                                      T* output_processed_3,
                                      T* output_processed_4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
            float a2 = output_1[i] + output_2[i] + output_3[i];
            float a5 = 0.5 * (a2 + output_4[i]);
            output_processed_1[i] = a5 - output_1[i] + output_5[i];
            output_processed_2[i] = a5 - a2 + output_6[i];
            output_processed_3[i] = a5 - output_2[i] + output_7[i];
            output_processed_4[i] = a5 - output_3[i] + output_8[i];
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct UpstrideInputFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* input_1,
                           const T* input_2,
                           const T* input_3,
                           const T* input_4,
                           T* input_processed_1,
                           T* input_processed_2,
                           T* input_processed_3,
                           T* input_processed_4,
                           T* input_processed_5,
                           T* input_processed_6,
                           T* input_processed_7,
                           T* input_processed_8) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = (size + 255) / 256;
    UpstrideInputCudaKernel<T>
        <<<block_count, 256, 0, d.stream()>>>(size, input_1, input_2, input_3, input_4,
                                                          input_processed_1,
                                                          input_processed_2,
                                                          input_processed_3,
                                                          input_processed_4,
                                                          input_processed_5,
                                                          input_processed_6,
                                                          input_processed_7,
                                                          input_processed_8);
  }
};

template <typename T>
struct UpstrideKernelFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* kernel_1,
                           const T* kernel_2,
                           const T* kernel_3,
                           const T* kernel_4,
                           T* kernel_processed_1,
                           T* kernel_processed_2,
                           T* kernel_processed_3,
                           T* kernel_processed_4,
                           T* kernel_processed_5,
                           T* kernel_processed_6,
                           T* kernel_processed_7,
                           T* kernel_processed_8) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = (size + 255) / 256;
    UpstrideKernelCudaKernel<T>
        <<<block_count, 256, 0, d.stream()>>>(size, kernel_1, kernel_2, kernel_3, kernel_4,
                                                          kernel_processed_1,
                                                          kernel_processed_2,
                                                          kernel_processed_3,
                                                          kernel_processed_4,
                                                          kernel_processed_5,
                                                          kernel_processed_6,
                                                          kernel_processed_7,
                                                          kernel_processed_8);
  }
};


template <typename T>
struct UpstrideOutputFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size,
                           const T* output_1,
                           const T* output_2,
                           const T* output_3,
                           const T* output_4,
                           const T* output_5,
                           const T* output_6,
                           const T* output_7,
                           const T* output_8,
                           T* output_processed_1,
                           T* output_processed_2,
                           T* output_processed_3,
                           T* output_processed_4) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = (size + 255) / 256;
    UpstrideOutputCudaKernel<T>
        <<<block_count, 256, 0, d.stream()>>>(size,
                                              output_1,
                                              output_2,
                                              output_3,
                                              output_4,
                                              output_5,
                                              output_6,
                                              output_7,
                                              output_8,
                                              output_processed_1,
                                              output_processed_2,
                                              output_processed_3,
                                              output_processed_4);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct UpstrideInputFunctor<GPUDevice, float>;
template struct UpstrideInputFunctor<GPUDevice, int32>;
template struct UpstrideKernelFunctor<GPUDevice, float>;
template struct UpstrideKernelFunctor<GPUDevice, int32>;
template struct UpstrideOutputFunctor<GPUDevice, float>;
template struct UpstrideOutputFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
