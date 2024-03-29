#pragma once
#include <stdexcept>

#include "tensorflow_includes.hpp"
#include "utils.hpp"

namespace upstride {
namespace frontend_tf {

/**
* @brief Convert a TensorShape into an upstride::Shape
* @param ts the input shape
* @return Shape upstride::Shape
*/
inline Shape toUpstrideShape(const tensorflow::TensorShape& ts) {
    Shape s(ts.dims());
    for (int i = 0; i < ts.dims(); ++i)
        s[i] = ts.dim_size(i);
    return s;
}

/**
 * @brief Converts upstride::Shape to a TensorShape
 * @param inShape the input shape
 * @return a corresponding tensorflow::TensorShape 
 */
inline tensorflow::TensorShape toTensorflowShape(const Shape& inShape) {
    tensorflow::TensorShape outShape;
    for (int i = 0; i < inShape.getSize(); ++i)
        outShape.AddDim(inShape[i]);
    return outShape;
}

/**
 * @brief Retrieves Algebra requested to be used for a given Op
 * Communicates an error to Tensorflow if the uptype of the Op cannot be interpreted or is undefined.
 * @param context   Kernel construction context for the Op
 * @return upstride::Algebra
 */
inline upstride::Algebra getAlgebra(tensorflow::OpKernelConstruction* context) {
    int uptype;
    if (context->GetAttr("uptype", &uptype) != tensorflow::Status::OK())
        context->CtxFailure(__FILE__, __LINE__, tensorflow::errors::Internal("uptype is not specified"));
    try {
        return upstride::getAlgebraFromType(uptype);
    } catch (std::invalid_argument& ex) {
        context->CtxFailure(__FILE__, __LINE__, tensorflow::errors::Internal(ex.what()));
    }

    return Algebra::REAL;
}

/**
 * @brief Computes output size along a single dimension of an operation that samples the input with strided/dilated patches.
 * Performs symbolic computation using TF routines mimicking the actual computation done in upstride::computeWindowedOutputSizeAndPadding()
 * @param ctx           TensorFlow inference context
 * @param inputSize     The input size
 * @param filterSize    The patch size
 * @param dilation      The patch dilation
 * @param stride        The patch stride
 * @param padding       Input padding preset
 * @param padBefore     Explicit zero padding at the beginning; only taken into account if the padding preset is EXPLICIT
 * @param padAfter      EXplicit zero padding at the end; only taken into account if the padding preset is EXPLICIT
 * @param outputSize    Resulting output size
 * @return TensorFlow Status
 */
inline tensorflow::Status computeWindowedOutputSize(
    tensorflow::shape_inference::InferenceContext* ctx,
    tensorflow::shape_inference::DimensionHandle inputSize,
    tensorflow::shape_inference::DimensionOrConstant filterSize,
    int dilation, int stride, upstride::Padding padding,
    int padBefore, int padAfter,
    tensorflow::shape_inference::DimensionHandle& outputSize) {
    if (stride <= 0)
        return tensorflow::errors::InvalidArgument("Stride must be > 0, but got ", stride);

    if (dilation < 1)
        return tensorflow::errors::InvalidArgument("Dilation rate must be >= 1, but got ", dilation);

    if (padding == upstride::Padding::VALID || padding == upstride::Padding::EXPLICIT) {
        // compute effective filter size
        tensorflow::shape_inference::DimensionHandle effectiveFilterSize;
        TF_RETURN_IF_ERROR(ctx->Subtract(ctx->MakeDim(filterSize), 1, &effectiveFilterSize));
        if (dilation != 1)
            TF_RETURN_IF_ERROR(ctx->Multiply(effectiveFilterSize, dilation, &effectiveFilterSize));
        TF_RETURN_IF_ERROR(ctx->Add(effectiveFilterSize, 1, &effectiveFilterSize));

        // compute the actual output size
        TF_RETURN_IF_ERROR(ctx->Subtract(inputSize, effectiveFilterSize, &outputSize));
        TF_RETURN_IF_ERROR(ctx->Add(outputSize,
                                    padding == upstride::Padding::EXPLICIT ? stride + padBefore + padAfter : stride,
                                    &outputSize));
        TF_RETURN_IF_ERROR(ctx->Divide(outputSize, stride, false, &outputSize));
    } else {
        TF_RETURN_IF_ERROR(ctx->Add(inputSize, stride - 1, &outputSize));
        TF_RETURN_IF_ERROR(ctx->Divide(outputSize, stride, false, &outputSize));
    }
    return tensorflow::Status::OK();
}

/**
 * @brief TensorFlow tensor representation
 * @tparam Device           The device type the tensor is stored on and used by
 * @tparam tf_type          TensorFlow datatype of tensor entries
 * @tparam core_type        Internally used (core) datatype of tensor entries
 */
template <typename Device, typename tf_type, typename core_type>
class InputTensorBase : public Tensor<Device, const core_type> {
   public:
    /**
     * @brief Construct a new Tensor object from a Tensorflow input Tensor
     * @param context   Tensorflow context
     * @param device    Device descriptor the tensor is stored on
     * @param idx       Index of the tensor to get in the context
     */
    InputTensorBase(tensorflow::OpKernelContext* context, Device& device, const int idx) :
        Tensor<Device, const core_type>(
            device, toUpstrideShape(context->input(idx).shape()),
            reinterpret_cast<const core_type*>(context->input(idx).flat<tf_type>().data())) {}
};

template <typename Device, typename T>
class InputTensorTF : public InputTensorBase<Device, T, T> {
    using InputTensorBase<Device, T, T>::InputTensorBase;
};

#ifdef BACKEND_CUDNN
template <>
class InputTensorTF<upstride::device::CUDA, upstride::cudnn::half>
    : public InputTensorBase<upstride::device::CUDA, Eigen::half, upstride::cudnn::half> {
    using InputTensorBase<upstride::device::CUDA, Eigen::half, upstride::cudnn::half>::InputTensorBase;
};
#endif

/**
 * @brief Representation of a writable TensorFlow tensor
 * @tparam Device           The device type the tensor is stored on and used by
 * @tparam tf_type          TensorFlow datatype of tensor entries
 * @tparam core_type        Internally used (core) datatype of tensor entries
 */
template <typename Device, typename tf_type, typename core_type>
class OutputTensorBase : public Tensor<Device, core_type> {
    static core_type* getOutputPtr(tensorflow::OpKernelContext* context, const tensorflow::TensorShape& shape,
                                       const int index) {
        tensorflow::Tensor* tensor = nullptr;
        ::tensorflow::Status status(context->allocate_output(index, shape, &tensor));
        if (!TF_PREDICT_TRUE(status.ok()))
            throw std::runtime_error("Cannot allocate output tensor");
        return reinterpret_cast<core_type*>(tensor->flat<tf_type>().data());
    }

   public:
    /**
     * @brief Wraps an output tensor of a Tensorflow operation in an upstride::tensor
     * @param context   Tensorflow OpKernel context
     * @param device    Device descriptor the tensor is stored on
     * @param shape     Output tensor shape
     * @param idx       Operation output index
     */
    OutputTensorBase(tensorflow::OpKernelContext* context, Device& device, const tensorflow::TensorShape& shape,
                       const int idx = 0) :
        Tensor<Device, core_type>(device, toUpstrideShape(shape), getOutputPtr(context, shape, idx)) {}
};

template <typename Device, typename T>
class OutputTensorTF : public OutputTensorBase<Device, T, T> {
    using OutputTensorBase<Device, T, T>::OutputTensorBase;
};

#ifdef BACKEND_CUDNN
template <>
class OutputTensorTF<upstride::device::CUDA, upstride::cudnn::half>
    : public OutputTensorBase<upstride::device::CUDA, Eigen::half, upstride::cudnn::half> {
    using OutputTensorBase<upstride::device::CUDA, Eigen::half, upstride::cudnn::half>::OutputTensorBase;
};
#endif

}  // namespace frontend_tf
}  // namespace upstride