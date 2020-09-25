#include "tensorflow_includes.hpp"
#include "upstride.hpp"
#include "upstride_tf.hpp"

namespace tensorflow {

/**
 * @brief Retrieves a Context instance.
 * @tparam Device   the device type
 * @return a context instance.
 */
template<typename Device>
inline upstride::Context& getContextInstance();

/**
 * @brief Retrieves a Device instance from an OpKernelContext instance.
 * @tparam Device   the device type
 * @param context   The OpKernelContext instance
 * @return the device instance.
 */
template<typename Device>
inline Device& fromTensorflowDevice(OpKernelContext* context);

template<>
inline upstride::Context& getContextInstance<upstride::device::CPU>() {
    static upstride::onednn::Context context;
    return context;
}

template<>
inline upstride::device::CPU& fromTensorflowDevice(OpKernelContext* context) {
    // nothing special to do here
    static upstride::device::CPU device;
    return device;
}

#ifdef BACKEND_CUDNN
template<>
inline upstride::Context& getContextInstance<upstride::device::CUDA>() {
    static upstride::cudnn::Context context;
    return context;
}

template<>
inline upstride::device::CUDA& fromTensorflowDevice(OpKernelContext* context) {
    auto stream = context->eigen_device<Eigen::GpuDevice>().stream();
    // CUDA devices are identified by their streams
    return static_cast<upstride::cudnn::Context&>(getContextInstance<upstride::device::CUDA>()).registerDevice(stream);
}
#endif

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class UpstrideConv2DOpKernel : public OpKernel, private upstride::UpstrideConv2DFunctor<Device, T> {
    static const int
        INPUT_IMAGE_IDX = 0,   //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 1,  //!< index of the input tensor containing the filter
        INPUT_BIAS_IDX = 2;    //!< index of the input tensor containing the bias

    const upstride::Algebra algebra;  //!< algebra to use within the Op
    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    upstride::IntTuple explicitPadding;
    upstride::IntTuple stride;
    upstride::IntTuple dilation;
    int groups;
    bool useBias;

   public:
    explicit UpstrideConv2DOpKernel(OpKernelConstruction* context) : OpKernel(context), upstride::UpstrideConv2DFunctor<Device, T>(getContextInstance<Device>()), algebra(upstride::frontend_tf::getAlgebra(context)) {
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
        OP_REQUIRES_OK(context, context->GetAttr("use_bias", &useBias));

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
            Device& device(fromTensorflowDevice<Device>(context));

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
            if (useBias) {
                InputTensorTF<Device, T> bias(context, device, INPUT_BIAS_IDX);
                (*this)(device, input, filter, &bias, output, padBefore, padAfter, groups);
            }
            else {
                (*this)(device, input, filter, nullptr, output, padBefore, padAfter, groups);
            }
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

    explicit UpstrideConv2DGradOpKernel(OpKernelConstruction* context) : OpKernel(context), upstride::UpstrideConv2DGradFunctor<Device, T>(getContextInstance<Device>()), algebra(upstride::frontend_tf::getAlgebra(context)) {
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
            Device& device(fromTensorflowDevice<Device>(context));

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
            (*this)(device, input, kernel, grad, kernelGrad, inputGrad, padBefore, padAfter, groups);
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};


/**
 * @brief A debugging/profiling op blocking until all the CUDA kernels are executed
 * Has no effect when run on CPU.
 */
template<class Device>
class UpstrideWaitOpKernel : public OpKernel {
   public:
    explicit UpstrideWaitOpKernel(OpKernelConstruction* context): OpKernel(context) {}

    void Compute(OpKernelContext* context) override;
};


#ifdef BACKEND_CUDNN
/**
 * @brief Specialization of Wait operation for a CUDA device.
 * Blocks till all the CUDA kernels in a CUDA stream associated with the device are executed.
 * @param context operation kernel context
 */
template<>
void UpstrideWaitOpKernel<upstride::device::CUDA>::Compute(OpKernelContext* context) {
    cudaStreamSynchronize(context->eigen_device<Eigen::GpuDevice>().stream());
}
#endif


/**
 * @brief Specialization of Wait operation for CPU.
 * Does nothing.
 */
template<>
void UpstrideWaitOpKernel<upstride::device::CPU>::Compute(OpKernelContext*) {}

/**
 * @brief Macro registering operations kernels
 * @param TF_TYPE       A TensorFlow datatype of input and output tensor entries
 * @param CORE_TYPE     Input and output tensor entries datatype used by the core
 * @param CPU_OR_GPU    "CPU" or "GPU" literally, specifying the frontend device the kernel is implemented for
 * @param DEVICE        Backend (core) device
 * @param OP_NAME       Op name
 */
#define REGISTER_UPSTRIDE_OP__FULL(TF_TYPE, CORE_TYPE, CPU_OR_GPU, DEVICE, OP_NAME)                  \
    REGISTER_KERNEL_BUILDER(Name(#OP_NAME).Device(DEVICE_##CPU_OR_GPU).TypeConstraint<TF_TYPE>("T"), \
                            OP_NAME##OpKernel<upstride::device::DEVICE, CORE_TYPE>)

/**
 * @brief A shortcut macro assuming same TF_TYPE and CORE_TYPE
 */
#define REGISTER_UPSTRIDE_OP(T, CPU_OR_GPU, DEVICE, OP_NAME) \
    REGISTER_UPSTRIDE_OP__FULL(T, T, CPU_OR_GPU, DEVICE, OP_NAME)

// Register the CPU kernels.
REGISTER_UPSTRIDE_OP(float, CPU, CPU, UpstrideConv2D);
REGISTER_UPSTRIDE_OP(float, CPU, CPU, UpstrideConv2DGrad);

REGISTER_KERNEL_BUILDER(Name("Wait").Device(DEVICE_CPU), UpstrideWaitOpKernel<upstride::device::CPU>);

// Register the GPU kernels.
#ifdef BACKEND_CUDNN
#ifdef UPSTRIDE_ENABLE_FP16
REGISTER_UPSTRIDE_OP__FULL(Eigen::half, upstride::cudnn::half, GPU, CUDA, UpstrideConv2D);
REGISTER_UPSTRIDE_OP__FULL(Eigen::half, upstride::cudnn::half, GPU, CUDA, UpstrideConv2DGrad);
#endif
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideConv2D);
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideConv2DGrad);

REGISTER_KERNEL_BUILDER(Name("Wait").Device(DEVICE_GPU), UpstrideWaitOpKernel<upstride::device::CUDA>);
#endif
}  // namespace tensorflow
