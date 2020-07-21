#pragma once
#include "tensorflow_includes.hpp"
#include "utils.hpp"

namespace upstride {
namespace frontend_tf {

/**
* @brief Convert a TensorShape into an upstride::Shape 
* 
* @param ts TensorShape
* @return Shape upstride::Shape
*/
Shape toUpstrideShape(const tensorflow::TensorShape& ts) {
    Shape s(ts.dims());
    for (int i = 0; i < ts.dims(); ++i) {
        s[i] = ts.dim_size(i);
    }
    return s;
}


tensorflow::TensorShape toTensorflowShape(const Shape& inShape) {
    tensorflow::TensorShape outShape;
    for (int i = 0; i < inShape.getSize(); ++i) {
        outShape.AddDim(inShape[i]);
    }
    return outShape;
}


template <typename T>
class InputTensorTF : public Tensor<const T> {
   public:
    /**
     * @brief Construct a new Tensor object from a Tensorflow input Tensor
     * 
     * @param context Tensorflow context
     * @param idx Index of the tensor to get in the context
     */
    InputTensorTF(tensorflow::OpKernelContext* context, const int idx) : Tensor<const T>(toUpstrideShape(context->input(idx).shape()),
                                                                                         context->input(idx).flat<T>().data()) {}
};


/**
 * @brief Tensorflow tensor representation inherit from upstride::tensor
 * 
 * @tparam T Tensorflow Tensor type
 */
template <typename T>
class OutputTensorTF : public Tensor<T> {
    static T* getOutputPtr(tensorflow::OpKernelContext* context, const tensorflow::TensorShape& shape, const int index) {
        tensorflow::Tensor* tensor = nullptr;
        ::tensorflow::Status status(context->allocate_output(index, shape, &tensor));
        if (!TF_PREDICT_TRUE(status.ok()))
            throw std::runtime_error("Cannot allocate output tensor");
        return tensor->flat<T>().data();
    }

   public:
    /**
     * @brief Wraps an output tensor of a Tensorflow operation in an upstride::tensor
     * 
     * @param context   Tensorflow OpKernel context
     * @param shape     Output tensor shape
     * @param idx       Operation output index
     */
    OutputTensorTF(tensorflow::OpKernelContext* context, const tensorflow::TensorShape& shape, const int idx = 0) : Tensor<T>(toUpstrideShape(shape),
                                                                                                                              getOutputPtr(context, shape, idx)) {}
};

}  // namespace frontend_tf
}  // namespace upstride