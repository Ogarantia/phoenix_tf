#pragma once
#include "tensorflow_includes.hpp"
#include "utils.hpp"

namespace upstride {
namespace tensorflow {

/**
* @brief Convert a TensorShape into an upstride::Shape 
* 
* @param ts TensorShape
* @return Shape upstride::Shape
*/
Shape convertShape(const TensorShape& ts) {
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
class TensorTF : protected Tensor<T> {
    /**
     * @brief Construct a new Tensor object from a Tensorflow Tensor
     * 
     * @param context Tensorflow context
     * @param idx Index of the tensor to get in the context
     */
    TensorTF(OpKernelContext* context, const int idx) : Tensor(computeShape(context->input(idx).shape()),
                                                               context->input(idx).flat<T>().data()) 
                                                    { }
};

}  // namespace tensorflow
}  // namespace upstride