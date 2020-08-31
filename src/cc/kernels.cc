#include "tensorflow_includes.hpp"
#include "upstride.hpp"
#include "upstride_tf.hpp"

namespace tensorflow {

template<typename Device>
inline const Device& fromTensorflowDevice(OpKernelContext* context);

template<>
inline const upstride::device::CPU& fromTensorflowDevice(OpKernelContext* context) {
    // nothing special to do here
    static upstride::device::CPU device;
    return device;
}

template<>
inline const upstride::device::CUDA& fromTensorflowDevice(OpKernelContext* context) {
    auto stream = context->eigen_device<Eigen::GpuDevice>().stream();
    // CUDA devices are identified by their streams
    return upstride::cudnn::Context::getInstance().registerDevice(stream);
}

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class UpstrideConv2DOpKernel : public OpKernel, private upstride::UpstrideConv2DFunctor<Device, T> {
    static const int
        INPUT_IMAGE_IDX = 0,   //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 1;  //!< index of the input tensor containing the filter

    const upstride::Algebra algebra;  //!< algebra to use within the Op
    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    upstride::IntTuple explicitPadding;
    upstride::IntTuple stride;
    upstride::IntTuple dilation;
    int groups;

   public:
    explicit UpstrideConv2DOpKernel(OpKernelConstruction* context) : OpKernel(context), algebra(upstride::frontend_tf::getAlgebra(context)) {
        // fetch parameters
        OP_REQUIRES_OK(context, context->GetAttr("strides", &stride));
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation));

        std::string paddingStr;
        OP_REQUIRES_OK(context, context->GetAttr("padding", &paddingStr));
        paddingPreset = upstride::paddingFromString(paddingStr);
        if (paddingPreset == upstride::Padding::EXPLICIT)
            OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings", &explicitPadding));

        std::string dataFormatStr;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &dataFormatStr));
        dataFormat = upstride::dataFormatFromString(dataFormatStr);

        OP_REQUIRES_OK(context, context->GetAttr("groups", &groups));

        // configure the operation backend
        //FIXME: Check status and throw an exception
        upstride::IntPair st, dil;
        upstride::getSpatialStep(stride, 1, st);
        upstride::getSpatialStep(dilation, 1, dil);

        upstride::UpstrideConv2DFunctor<Device, T>::configure(algebra, dataFormat, st, dil);
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            const Device& device(fromTensorflowDevice<Device>(context));

            // grab inputs
            InputTensorTF<Device, T> input(context, device, INPUT_IMAGE_IDX);
            InputTensorTF<Device, T> filter(context, device, INPUT_FILTER_IDX);

            // compute output shape and paddings
            upstride::IntPair padBefore, padAfter;
            TensorShape outShape = toTensorflowShape(upstride::computeConvOutputSize(
                algebra, dataFormat,
                input.getShape(), filter.getShape(),
                paddingPreset, explicitPadding, stride, dilation, padBefore, padAfter, groups));

            // allocate output tensor
            OutputTensorTF<Device, T> output(context, device, outShape);

            // execute the operation
            (*this)(input, filter, output, padBefore, padAfter, groups);
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};

template <typename Device, typename T>
class UpstrideConv2DGradOpKernel : public OpKernel, private upstride::UpstrideConv2DGradFunctor<Device, T> {
    const upstride::Algebra algebra;  //!< algebra to use within the Op
    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    upstride::IntTuple explicitPadding;
    upstride::IntTuple stride;
    upstride::IntTuple dilation;
    int groups;
    bool requireInputGrad;

   public:
    static const int
        INPUT_GRAD_IDX = 0,    //!< index of the input tensor containing the loss function gradient
        INPUT_INPUT_IDX = 1,   //!< index of the input tensor containing the image
        INPUT_KERNEL_IDX = 2;  //!< index of the input tensor containing the filter
    static const int
        OUTPUT_KERNELGRAD_IDX = 1,  //!< index of the output tensor containing the loss function gradient
        OUPUT_INPUTGRAD_IDX = 0;    //!< index of the output tensor containing the filter

    explicit UpstrideConv2DGradOpKernel(OpKernelConstruction* context) : OpKernel(context), algebra(upstride::frontend_tf::getAlgebra(context)) {
        // fetch parameters
        OP_REQUIRES_OK(context, context->GetAttr("strides", &stride));
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation));

        std::string paddingStr;
        OP_REQUIRES_OK(context, context->GetAttr("padding", &paddingStr));
        paddingPreset = upstride::paddingFromString(paddingStr);
        if (paddingPreset == upstride::Padding::EXPLICIT)
            OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings", &explicitPadding));

        std::string dataFormatStr;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &dataFormatStr));
        dataFormat = upstride::dataFormatFromString(dataFormatStr);

        OP_REQUIRES_OK(context, context->GetAttr("require_input_grad", &requireInputGrad));

        OP_REQUIRES_OK(context, context->GetAttr("groups", &groups));
        // configure the operation backend
        try {
            upstride::IntPair st, dil;
            upstride::getSpatialStep(stride, 1, st);
            upstride::getSpatialStep(dilation, 1, dil);
            upstride::UpstrideConv2DGradFunctor<Device, T>::configure(algebra, dataFormat, st, dil, requireInputGrad);
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            const Device& device(fromTensorflowDevice<Device>(context));

            // grab inputs
            InputTensorTF<Device, T> grad(context, device, INPUT_GRAD_IDX);
            InputTensorTF<Device, T> kernel(context, device, INPUT_KERNEL_IDX);
            InputTensorTF<Device, T> input(context, device, INPUT_INPUT_IDX);

            // compute output shape and paddings
            upstride::IntPair padBefore, padAfter;
            TensorShape outShape = toTensorflowShape(upstride::computeConvOutputSize(
                algebra, dataFormat,
                input.getShape(), kernel.getShape(),
                paddingPreset, explicitPadding, stride, dilation, padBefore, padAfter, groups));

            // allocate output tensor
            OutputTensorTF<Device, T> kernelGrad(context, device, context->input(INPUT_KERNEL_IDX).shape(), OUTPUT_KERNELGRAD_IDX);
            OutputTensorTF<Device, T> inputGrad(context, device, context->input(INPUT_INPUT_IDX).shape(), OUPUT_INPUTGRAD_IDX);

            // execute the operation
            (*this)(input, kernel, grad, kernelGrad, inputGrad, padBefore, padAfter, groups);
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};

// Register the CPU kernels.
#define REGISTER_UPSTRIDE_OP(T, CPU_OR_GPU, DEVICE, OP_NAME)               \
    REGISTER_KERNEL_BUILDER(                                               \
        Name(#OP_NAME).Device(DEVICE_##CPU_OR_GPU).TypeConstraint<T>("T"), \
        OP_NAME##OpKernel<upstride::device::DEVICE, T>);

// Register the CPU kernels.
REGISTER_UPSTRIDE_OP(float, CPU, CPU, UpstrideConv2D);
REGISTER_UPSTRIDE_OP(float, CPU, CPU, UpstrideConv2DGrad);

// Register the GPU kernels.
#ifdef BACKEND_CUDNN
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideConv2D);
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideConv2DGrad);
#endif
}  // namespace tensorflow
