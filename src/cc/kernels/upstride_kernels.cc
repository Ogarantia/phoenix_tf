#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "upstride.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T>
struct UpstrideInputFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d, int size,
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
        for (int i = 0; i < size; ++i) {
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
};

template <typename T>
struct UpstrideKernelFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d, int size,
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
        for (int i = 0; i < size; ++i) {
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
};

template <typename T>
struct UpstrideOutputFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d, int size,
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
        for (int i = 0; i < size; ++i) {
            float a2 = output_1[i] + output_2[i] + output_3[i];
            float a5 = 0.5 * (a2 + output_4[i]);
            output_processed_1[i] = a5 - output_1[i] + output_5[i];
            output_processed_2[i] = a5 - a2 + output_6[i];
            output_processed_3[i] = a5 - output_2[i] + output_7[i];
            output_processed_4[i] = a5 - output_3[i] + output_8[i];
        }
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class UpstrideInputOp : public OpKernel {
   public:
    explicit UpstrideInputOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& input_tensor_0 = context->input(0);
        const Tensor& input_tensor_1 = context->input(1);
        const Tensor& input_tensor_2 = context->input(2);
        const Tensor& input_tensor_3 = context->input(3);

        // Create output tensors
        Tensor* output_tensor_0 = NULL;
        Tensor* output_tensor_1 = NULL;
        Tensor* output_tensor_2 = NULL;
        Tensor* output_tensor_3 = NULL;
        Tensor* output_tensor_4 = NULL;
        Tensor* output_tensor_5 = NULL;
        Tensor* output_tensor_6 = NULL;
        Tensor* output_tensor_7 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_0.shape(), &output_tensor_0));
        OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor_0.shape(), &output_tensor_1));
        OP_REQUIRES_OK(context, context->allocate_output(2, input_tensor_0.shape(), &output_tensor_2));
        OP_REQUIRES_OK(context, context->allocate_output(3, input_tensor_0.shape(), &output_tensor_3));
        OP_REQUIRES_OK(context, context->allocate_output(4, input_tensor_0.shape(), &output_tensor_4));
        OP_REQUIRES_OK(context, context->allocate_output(5, input_tensor_0.shape(), &output_tensor_5));
        OP_REQUIRES_OK(context, context->allocate_output(6, input_tensor_0.shape(), &output_tensor_6));
        OP_REQUIRES_OK(context, context->allocate_output(7, input_tensor_0.shape(), &output_tensor_7));

        // Do the computation.
        OP_REQUIRES(context, input_tensor_0.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        UpstrideInputFunctor<Device, T>()(
            context->eigen_device<Device>(),
            static_cast<int>(input_tensor_0.NumElements()),
            input_tensor_0.flat<T>().data(),
            input_tensor_1.flat<T>().data(),
            input_tensor_2.flat<T>().data(),
            input_tensor_3.flat<T>().data(),
            output_tensor_0->flat<T>().data(),
            output_tensor_1->flat<T>().data(),
            output_tensor_2->flat<T>().data(),
            output_tensor_3->flat<T>().data(),
            output_tensor_4->flat<T>().data(),
            output_tensor_5->flat<T>().data(),
            output_tensor_6->flat<T>().data(),
            output_tensor_7->flat<T>().data());
    }
};

template <typename Device, typename T>
class UpstrideKernelOp : public OpKernel {
   public:
    explicit UpstrideKernelOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& input_tensor_0 = context->input(0);
        const Tensor& input_tensor_1 = context->input(1);
        const Tensor& input_tensor_2 = context->input(2);
        const Tensor& input_tensor_3 = context->input(3);

        // Create output tensors
        Tensor* output_tensor_0 = NULL;
        Tensor* output_tensor_1 = NULL;
        Tensor* output_tensor_2 = NULL;
        Tensor* output_tensor_3 = NULL;
        Tensor* output_tensor_4 = NULL;
        Tensor* output_tensor_5 = NULL;
        Tensor* output_tensor_6 = NULL;
        Tensor* output_tensor_7 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_0.shape(), &output_tensor_0));
        OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor_0.shape(), &output_tensor_1));
        OP_REQUIRES_OK(context, context->allocate_output(2, input_tensor_0.shape(), &output_tensor_2));
        OP_REQUIRES_OK(context, context->allocate_output(3, input_tensor_0.shape(), &output_tensor_3));
        OP_REQUIRES_OK(context, context->allocate_output(4, input_tensor_0.shape(), &output_tensor_4));
        OP_REQUIRES_OK(context, context->allocate_output(5, input_tensor_0.shape(), &output_tensor_5));
        OP_REQUIRES_OK(context, context->allocate_output(6, input_tensor_0.shape(), &output_tensor_6));
        OP_REQUIRES_OK(context, context->allocate_output(7, input_tensor_0.shape(), &output_tensor_7));

        // Do the computation.
        OP_REQUIRES(context, input_tensor_0.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        UpstrideKernelFunctor<Device, T>()(
            context->eigen_device<Device>(),
            static_cast<int>(input_tensor_0.NumElements()),
            input_tensor_0.flat<T>().data(),
            input_tensor_1.flat<T>().data(),
            input_tensor_2.flat<T>().data(),
            input_tensor_3.flat<T>().data(),
            output_tensor_0->flat<T>().data(),
            output_tensor_1->flat<T>().data(),
            output_tensor_2->flat<T>().data(),
            output_tensor_3->flat<T>().data(),
            output_tensor_4->flat<T>().data(),
            output_tensor_5->flat<T>().data(),
            output_tensor_6->flat<T>().data(),
            output_tensor_7->flat<T>().data());
    }
};

template <typename Device, typename T>
class UpstrideOutputOp : public OpKernel {
   public:
    explicit UpstrideOutputOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& input_tensor_0 = context->input(0);
        const Tensor& input_tensor_1 = context->input(1);
        const Tensor& input_tensor_2 = context->input(2);
        const Tensor& input_tensor_3 = context->input(3);
        const Tensor& input_tensor_4 = context->input(4);
        const Tensor& input_tensor_5 = context->input(5);
        const Tensor& input_tensor_6 = context->input(6);
        const Tensor& input_tensor_7 = context->input(7);

        // Create output tensors
        Tensor* output_tensor_0 = NULL;
        Tensor* output_tensor_1 = NULL;
        Tensor* output_tensor_2 = NULL;
        Tensor* output_tensor_3 = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_0.shape(), &output_tensor_0));
        OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor_0.shape(), &output_tensor_1));
        OP_REQUIRES_OK(context, context->allocate_output(2, input_tensor_0.shape(), &output_tensor_2));
        OP_REQUIRES_OK(context, context->allocate_output(3, input_tensor_0.shape(), &output_tensor_3));

        // Do the computation.
        OP_REQUIRES(context, input_tensor_0.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        UpstrideOutputFunctor<Device, T>()(
            context->eigen_device<Device>(),
            static_cast<int>(input_tensor_0.NumElements()),
            input_tensor_0.flat<T>().data(),
            input_tensor_1.flat<T>().data(),
            input_tensor_2.flat<T>().data(),
            input_tensor_3.flat<T>().data(),
            input_tensor_4.flat<T>().data(),
            input_tensor_5.flat<T>().data(),
            input_tensor_6.flat<T>().data(),
            input_tensor_7.flat<T>().data(),
            output_tensor_0->flat<T>().data(),
            output_tensor_1->flat<T>().data(),
            output_tensor_2->flat<T>().data(),
            output_tensor_3->flat<T>().data());
    }
};

// Register the CPU kernels.
#define REGISTER_CPU_INPUT(T)                                            \
    REGISTER_KERNEL_BUILDER(                                             \
        Name("UpstrideInput").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        UpstrideInputOp<CPUDevice, T>);
REGISTER_CPU_INPUT(float);
REGISTER_CPU_INPUT(int32);

#define REGISTER_CPU_KERNEL(T)                                            \
    REGISTER_KERNEL_BUILDER(                                              \
        Name("UpstrideKernel").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        UpstrideKernelOp<CPUDevice, T>);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(int32);

#define REGISTER_CPU_OUTPUT(T)                                            \
    REGISTER_KERNEL_BUILDER(                                              \
        Name("UpstrideOutput").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
        UpstrideOutputOp<CPUDevice, T>);
REGISTER_CPU_OUTPUT(float);
REGISTER_CPU_OUTPUT(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_INPUT(T)                                            \
    extern template struct UpstrideInputFunctor<GPUDevice, T>;           \
    REGISTER_KERNEL_BUILDER(                                             \
        Name("UpstrideInput").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        UpstrideInputOp<GPUDevice, T>);
REGISTER_GPU_INPUT(float);
REGISTER_GPU_INPUT(int32);

#define REGISTER_GPU_KERNEL(T)                                            \
    extern template struct UpstrideKernelFunctor<GPUDevice, T>;           \
    REGISTER_KERNEL_BUILDER(                                              \
        Name("UpstrideKernel").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        UpstrideKernelOp<GPUDevice, T>);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(int32);

#define REGISTER_GPU_OUTPUT(T)                                            \
    extern template struct UpstrideOutputFunctor<GPUDevice, T>;           \
    REGISTER_KERNEL_BUILDER(                                              \
        Name("UpstrideOutput").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        UpstrideOutputOp<GPUDevice, T>);
REGISTER_GPU_OUTPUT(float);
REGISTER_GPU_OUTPUT(int32);
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace tensorflow
