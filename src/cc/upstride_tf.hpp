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
Shape convertShape(const tensorflow::TensorShape& ts) {
    Shape s(ts.dims());
    for (int i = 0; i < ts.dims(); ++i) {
        s[i] = ts.dim_size(i);
    }
    return s;
}

/**
 * @brief Tensorflow tensor representation inherit from upstride::tensor
 * 
 * @tparam T Tensorflow Tensor type
 */
template <typename T>
class TensorTF : public Tensor<T> {
    static T* getOutputPtr(tensorflow::OpKernelContext* context, const tensorflow::TensorShape& shape, const int index) {
        tensorflow::Tensor* tensor = nullptr;
        //OP_REQUIRES_OK(context, context->allocate_output(index, shape, &tensor));
        context->allocate_output(index, shape, &tensor);
        return tensor->flat<T>().data();
    }

   public:
    /**
     * @brief Construct a new Tensor object from a Tensorflow input Tensor
     * 
     * @param context Tensorflow context
     * @param idx Index of the tensor to get in the context
     */
    TensorTF(tensorflow::OpKernelContext* context, const int idx) : Tensor<T>(convertShape(context->input(idx).shape()),
                                                               context->input(idx).flat<T>().data()) {}

    /**
     * @brief Wraps an output tensor of a Tensorflow operation in an upstride::tensor
     * 
     * @param context   Tensorflow OpKernel context
     * @param shape     Output tensor shape
     * @param idx       Operation output index
     */
    TensorTF(tensorflow::OpKernelContext* context, const tensorflow::TensorShape& shape, const int idx = 0) : Tensor<T>(convertShape(shape),
                                                                                                     getOutputPtr(context, shape, idx)) {}
};

}  // namespace tensorflow
}  // namespace upstride