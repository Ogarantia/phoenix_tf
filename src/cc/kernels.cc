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
    static upstride::device::CPU device(getContextInstance<upstride::device::CPU>());
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
    auto referenceStream = context->eigen_device<Eigen::GpuDevice>().stream();
    // CUDA devices are identified by their streams
    return static_cast<upstride::cudnn::Context&>(getContextInstance<upstride::device::CUDA>()).registerDevice(referenceStream);
}
#endif

/**
 * @brief Temporary memory allocator callback implementation
 */
class TFAllocator : public upstride::Allocator {
private:
    const size_t alignmentConstraint;
    OpKernelContext* context;
    tensorflow::Tensor buffer;
    bool isAllocated;

public:
    template <typename Device>
    inline TFAllocator(OpKernelContext* context, Device& device):
        alignmentConstraint(device.getAlignmentConstraint()), context(context), isAllocated(false)
    {}

    inline void* mallocTemp(size_t size) override {
        if (isAllocated)
            throw std::runtime_error("Temporary memory already allocated");
        if (context->allocate_temp(DT_UINT8, { (int64)size }, &buffer) != ::tensorflow::Status::OK())
            throw std::runtime_error("Cannot allocate a temporary buffer of " + std::to_string(size) + " bytes");
        isAllocated = true;
        return buffer.flat<uint8_t>().data();
    }

    inline size_t getAlignmentConstraint() const override {
        return alignmentConstraint;
    }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class UpstrideConv2DOpKernel : public OpKernel {
    static const int
        INPUT_IMAGE_IDX = 0,   //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 1,  //!< index of the input tensor containing the filter
        INPUT_BIAS_IDX = 2;    //!< index of the input tensor containing the bias

    const upstride::Algebra algebra;  //!< algebra to use within the Op
    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    upstride::IntTuple explicitPadding;
    upstride::IntPair stride;
    upstride::IntPair dilation;
    int groups;
    bool useBias;
    bool realValuedInput;           //!< if `true`, the input of this Conv2D is real-valued

   public:
    explicit UpstrideConv2DOpKernel(OpKernelConstruction* context) : OpKernel(context),
                                                                     algebra(upstride::frontend_tf::getAlgebra(context)) {
        // fetch parameters
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
        OP_REQUIRES_OK(context, context->GetAttr("type0_inputs", &realValuedInput));

        // configure the operation backend
        //FIXME: Check status and throw an exception
        upstride::IntTuple stride, dilation;
        OP_REQUIRES_OK(context, context->GetAttr("strides", &stride));
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation));
        upstride::getSpatialStep(stride, 1, this->stride);
        upstride::getSpatialStep(dilation, 1, this->dilation);
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            Device& device(fromTensorflowDevice<Device>(context));

            // grab inputs
            InputTensorTF<Device, T> input(context, device, INPUT_IMAGE_IDX);
            InputTensorTF<Device, T> filter(context, device, INPUT_FILTER_IDX);

            // initialize a descriptor
            const upstride::Conv2DFwdDescriptor descriptor(
                input.getShape(), filter.getShape(), stride, dilation, paddingPreset, explicitPadding, groups, algebra, dataFormat, useBias, realValuedInput);

            // allocate output tensor
            OutputTensorTF<Device, T> output(context, device, toTensorflowShape(descriptor.getOutputShape()));

            // create an allocator instance
            TFAllocator allocator(context, device);

            // execute the operation
            if (useBias) {
                InputTensorTF<Device, T> bias(context, device, INPUT_BIAS_IDX);
                upstride::conv2DFwd<Device, T>(device, allocator, input, filter, &bias, output, descriptor);
            }
            else {
                upstride::conv2DFwd<Device, T>(device, allocator, input, filter, nullptr, output, descriptor);
            }
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};


template <typename Device, typename T>
class UpstrideConv2DGradOpKernel : public OpKernel {
    const upstride::Algebra algebra;  //!< algebra to use within the Op
    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    upstride::IntTuple explicitPadding;
    upstride::IntPair stride;
    upstride::IntPair dilation;
    int groups;
    bool requireInputGrad;
    bool realValuedInput;           //!< if `true`, the input of this Conv2D is real-valued

   public:
    static const int
        INPUT_GRAD_IDX = 0,         //!< index of the input tensor containing the loss function gradient
        INPUT_INPUT_IDX = 1,        //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 2;       //!< index of the input tensor containing the filter
    static const int
        OUPUT_FILTERGRAD_IDX = 1,  //!< index of the output tensor containing the loss function gradient
        OUPUT_INPUTGRAD_IDX = 0;    //!< index of the output tensor containing the filter

    explicit UpstrideConv2DGradOpKernel(OpKernelConstruction* context) : OpKernel(context),
                                                                         algebra(upstride::frontend_tf::getAlgebra(context)) {
        // fetch parameters
        std::string paddingStr;
        OP_REQUIRES_OK(context, context->GetAttr("padding", &paddingStr));
        paddingPreset = upstride::paddingFromString(paddingStr);
        if (paddingPreset == upstride::Padding::EXPLICIT)
            OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings", &explicitPadding));

        std::string dataFormatStr;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &dataFormatStr));
        dataFormat = upstride::dataFormatFromString(dataFormatStr);

        OP_REQUIRES_OK(context, context->GetAttr("require_input_grad", &requireInputGrad));
        OP_REQUIRES_OK(context, context->GetAttr("type0_inputs", &realValuedInput));
        OP_REQUIRES_OK(context, context->GetAttr("groups", &groups));

        upstride::IntTuple stride, dilation;
        OP_REQUIRES_OK(context, context->GetAttr("strides", &stride));
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation));
        upstride::getSpatialStep(stride, 1, this->stride);
        upstride::getSpatialStep(dilation, 1, this->dilation);
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            Device& device(fromTensorflowDevice<Device>(context));

            // grab inputs
            InputTensorTF<Device, T> grad(context, device, INPUT_GRAD_IDX);
            InputTensorTF<Device, T> filter(context, device, INPUT_FILTER_IDX);
            InputTensorTF<Device, T> input(context, device, INPUT_INPUT_IDX);

            // initialize a descriptor
            const upstride::Conv2DBwdDescriptor descriptor(
                input.getShape(), filter.getShape(), stride, dilation, paddingPreset, explicitPadding, groups, algebra, dataFormat, requireInputGrad, realValuedInput);

            // allocate output tensor
            OutputTensorTF<Device, T> filterGrad(context, device, context->input(INPUT_FILTER_IDX).shape(), OUPUT_FILTERGRAD_IDX);
            OutputTensorTF<Device, T> inputGrad(context, device, context->input(INPUT_INPUT_IDX).shape(), OUPUT_INPUTGRAD_IDX);

            // create an allocator instance
            TFAllocator allocator(context, device);

            // execute the operation
            upstride::conv2DBwd<Device, T>(device, allocator, input, filter, grad, filterGrad, inputGrad, descriptor);
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};


template <typename Device, typename T>
class UpstrideDenseOpKernel : public OpKernel {
    static const int
        INPUT_IMAGE_IDX = 0,   //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 1,  //!< index of the input tensor containing the filter
        INPUT_BIAS_IDX = 2;    //!< index of the input tensor containing the bias

    const upstride::Algebra algebra;  //!< algebra to use within the Op
    bool useBias;

   public:
    explicit UpstrideDenseOpKernel(OpKernelConstruction* context) : OpKernel(context),
                                                                    algebra(upstride::frontend_tf::getAlgebra(context)) {
        // fetch parameters
        OP_REQUIRES_OK(context, context->GetAttr("use_bias", &useBias));
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            Device& device(fromTensorflowDevice<Device>(context));
            // grab inputs
            InputTensorTF<Device, T> input(context, device, INPUT_IMAGE_IDX);
            InputTensorTF<Device, T> filter(context, device, INPUT_FILTER_IDX);

            // compute output shape
            TensorShape outShape = toTensorflowShape({input.getShape()[0], filter.getShape().getSize() == 3 ? filter.getShape()[2] : filter.getShape()[1]});

            // allocate output tensor
            OutputTensorTF<Device, T> output(context, device, outShape);

            // setup descriptor
            const upstride::DenseFwdDescriptor descriptor(input.getShape(), filter.getShape(), algebra, upstride::DataFormat::IO, useBias);

            // create an allocator instance
            TFAllocator allocator(context, device);

            // execute the operation
            if (useBias) {
                InputTensorTF<Device, T> bias(context, device, INPUT_BIAS_IDX);
                upstride::denseFwd<Device, T>(device, allocator, input, filter, &bias, output, descriptor);
            }
            else {
                upstride::denseFwd<Device, T>(device, allocator, input, filter, nullptr, output, descriptor);
            }
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};


template <typename Device, typename T>
class UpstrideDenseGradOpKernel : public OpKernel {
    const upstride::Algebra algebra;  //!< algebra to use within the Op
    bool requireInputGrad;
   public:
    static const int
        INPUT_GRAD_IDX = 0,    //!< index of the input tensor containing the loss function gradient
        INPUT_INPUT_IDX = 1,   //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 2;  //!< index of the input tensor containing the filter
    static const int
        OUPUT_FILTERGRAD_IDX = 1,  //!< index of the output tensor containing the loss function gradient
        OUPUT_INPUTGRAD_IDX = 0;    //!< index of the output tensor containing the filter

    explicit UpstrideDenseGradOpKernel(OpKernelConstruction* context) : OpKernel(context),
                                                                        algebra(upstride::frontend_tf::getAlgebra(context)) {
        // fetch parameters
        OP_REQUIRES_OK(context, context->GetAttr("require_input_grad", &requireInputGrad));
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            Device& device(fromTensorflowDevice<Device>(context));

            // grab inputs
            InputTensorTF<Device, T> grad(context, device, INPUT_GRAD_IDX);
            InputTensorTF<Device, T> filter(context, device, INPUT_FILTER_IDX);
            InputTensorTF<Device, T> input(context, device, INPUT_INPUT_IDX);

            // allocate output tensor
            OutputTensorTF<Device, T> filterGrad(context, device, context->input(INPUT_FILTER_IDX).shape(), OUPUT_FILTERGRAD_IDX);
            OutputTensorTF<Device, T> inputGrad(context, device, context->input(INPUT_INPUT_IDX).shape(), OUPUT_INPUTGRAD_IDX);

            // create an allocator instance
            TFAllocator allocator(context, device);

            // execute the operation
            const upstride::DenseBwdDescriptor descriptor(input.getShape(), filter.getShape(), algebra, upstride::DataFormat::IO, requireInputGrad);
            upstride::denseBwd<Device, T>(device, allocator, input, filter, grad, filterGrad, inputGrad, descriptor);
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


/**
 * @brief Recycles all the resources used by the engine.
 */
class UpstrideCleanUpOpKernel : public OpKernel {
   public:
    explicit UpstrideCleanUpOpKernel(OpKernelConstruction* context): OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        getContextInstance<upstride::device::CPU>().cleanUp();
#ifdef BACKEND_CUDNN
        getContextInstance<upstride::device::CUDA>().cleanUp();
#endif
    }
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
REGISTER_UPSTRIDE_OP(float, CPU, CPU, UpstrideDense);
REGISTER_UPSTRIDE_OP(float, CPU, CPU, UpstrideDenseGrad);
REGISTER_KERNEL_BUILDER(Name("Wait").Device(DEVICE_CPU), UpstrideWaitOpKernel<upstride::device::CPU>);
REGISTER_KERNEL_BUILDER(Name("CleanUp").Device(DEVICE_CPU), UpstrideCleanUpOpKernel);

// Register the GPU kernels.
#ifdef BACKEND_CUDNN
#ifdef UPSTRIDE_ENABLE_FP16
REGISTER_UPSTRIDE_OP__FULL(Eigen::half, upstride::cudnn::half, GPU, CUDA, UpstrideConv2D);
REGISTER_UPSTRIDE_OP__FULL(Eigen::half, upstride::cudnn::half, GPU, CUDA, UpstrideConv2DGrad);
REGISTER_UPSTRIDE_OP__FULL(Eigen::half, upstride::cudnn::half, GPU, CUDA, UpstrideDense);
REGISTER_UPSTRIDE_OP__FULL(Eigen::half, upstride::cudnn::half, GPU, CUDA, UpstrideDenseGrad);
#endif
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideConv2D);
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideConv2DGrad);
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideDense);
REGISTER_UPSTRIDE_OP(float, GPU, CUDA, UpstrideDenseGrad);
REGISTER_KERNEL_BUILDER(Name("Wait").Device(DEVICE_GPU), UpstrideWaitOpKernel<upstride::device::CUDA>);
#endif
}  // namespace tensorflow
