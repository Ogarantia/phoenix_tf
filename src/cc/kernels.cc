#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_includes.hpp"
#include "upstride.hpp"
#include "upstride_tf.hpp"

namespace tensorflow {

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class UpstrideConv2DOpKernel : public OpKernel, private upstride::UpstrideConv2DFunctor<Device, T> {
    static const int
        INPUT_IMAGE_IDX = 0,   //!< index of the input tensor containing the image
        INPUT_FILTER_IDX = 1;  //!< index of the input tensor containing the filter

    upstride::DataFormat dataFormat;
    upstride::Padding paddingPreset;
    std::vector<int32> explicitPadding;
    std::vector<int32> stride;
    std::vector<int32> dilation;

   public:
    explicit UpstrideConv2DOpKernel(OpKernelConstruction* context) : OpKernel(context) {
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
    }

    void Compute(OpKernelContext* context) override {
        using namespace upstride::frontend_tf;

        try {
            // grab inputs
            InputTensorTF<T> input(context, INPUT_IMAGE_IDX);
            InputTensorTF<T> filter(context, INPUT_FILTER_IDX);
            
            std::cout << "inputShape = " << input.getShape() << std::endl;
            std::cout << "filterShape = " << filter.getShape() << std::endl;

            // compute output shape
            TensorShape outShape = toTensorflowShape(upstride::computeConvOutputSize(
                1, dataFormat,
                input.getShape(), filter.getShape(),
                paddingPreset, explicitPadding, stride, dilation));

            // allocate output tensor
            OutputTensorTF<T> output(context, outShape);

            // execute the operation
            (*this)(input, filter, output, dataFormat);
        } catch (std::exception& ex) {
            context->CtxFailure(__FILE__, __LINE__, errors::Internal(ex.what()));
        }
    }
};

// Register the CPU kernels.
#define REGISTER_UPSTRIDE_OP(T, CPU_OR_GPU, OP_NAME)                       \
    REGISTER_KERNEL_BUILDER(                                               \
        Name(#OP_NAME).Device(DEVICE_##CPU_OR_GPU).TypeConstraint<T>("T"), \
        OP_NAME##OpKernel<upstride::device::CPU_OR_GPU, T>);
REGISTER_UPSTRIDE_OP(float, CPU, UpstrideConv2D);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_UPSTRIDE_OP(float, GPU, UpstrideConv2D);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
