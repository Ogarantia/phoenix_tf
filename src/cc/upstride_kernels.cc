#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_includes.hpp"
#include "upstride.hpp"
#include "upstride_tf.hpp"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

}  // namespace tensorflow

namespace upstride {
// CPU specialization of actual computation.
template <typename T>
struct UpstrideConv2DFunctor<tensorflow::CPUDevice, T> {
    void operator()(const Tensor<const T>& input,
                    const Tensor<const T>& kernel,
                    Tensor<T>& output) {
        //call oneDNN
        std::cout << " coucou c'est nous aussi!!! " << std::endl;
    }
};
}  // namespace upstride

namespace tensorflow {

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class UpstrideConv2DOpKernel : public OpKernel {
    static const int
        INPUT_IMAGE_IDX = 0,   //!< index of the input tensor containing the image
        INPUT_KERNEL_IDX = 1;  //!< index of the input tensor containing the convolution kernel

    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    upstride::IntPair padding;
    upstride::IntPair stride;
    upstride::IntPair dilation;

   public:
    explicit UpstrideConv2DOpKernel(OpKernelConstruction* context) : OpKernel(context) {
        // fetch parameters
        context->GetAttr<int>("strides", stride);
        context->GetAttr<int>("dilations", dilation);
        if (context->HasAttr("explicit_paddings"))
            context->GetAttr<int>("explicit_paddings", padding);

        std::string paddingStr;
        context->GetAttr("padding", &paddingStr);
        paddingPreset =  upstride::paddingFromString(paddingStr);

        std::string dataFormatStr;
        context->GetAttr("data_format", &dataFormatStr);
        dataFormat = upstride::dataFormatFromString(dataFormatStr);
    }

    void Compute(OpKernelContext* context) override {
        std::cout << " coucou c'est nous " << std::endl;

        using namespace upstride::frontend_tf;
        InputTensorTF<T> input(context, INPUT_IMAGE_IDX);
        InputTensorTF<T> kernel(context, INPUT_KERNEL_IDX);

        // compute output shape
        TensorShape outShape = toTensorflowShape(upstride::computeConvOutputSize(
            4, dataFormat,
            input.getShape(), kernel.getShape(),
            paddingPreset, padding, stride, dilation
        ));

        // allocate output tensor
        OutputTensorTF<T> output(context, outShape);

        upstride::UpstrideConv2DFunctor<Device, T>()(input, kernel, output);
    }
};

// Register the CPU kernels.
#define REGISTER_UPSTRIDE_OP(T, CPU_OR_GPU, OP_NAME)                       \
    REGISTER_KERNEL_BUILDER(                                               \
        Name(#OP_NAME).Device(DEVICE_##CPU_OR_GPU).TypeConstraint<T>("T"), \
        OP_NAME##OpKernel<CPU_OR_GPU##Device, T>);
REGISTER_UPSTRIDE_OP(float, CPU, UpstrideConv2D);
//REGISTER_UPSTRIDE_OP(int32, CPU, UpstrideConv2D);

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
}  // namespace tensorflow
