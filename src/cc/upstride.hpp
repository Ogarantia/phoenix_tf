#ifndef KERNEL_UPSTRIDE_OPS_H_
#define KERNEL_UPSTRIDE_OPS_H_

#include "utils.hpp"

namespace upstride {

namespace device {

typedef struct {
} CPU;
typedef struct {
} GPU;

}  // namespace device

template <typename Device, typename T>
struct UpstrideConv2DFunctor {
    /**
     * @brief 
     * 
     * @param input 
     * @param kernel 
     * @param output 
     */
    void operator()(const Tensor<const T>& input,
                    const Tensor<const T>& kernel,
                    Tensor<T>& output);
};

template<>
struct UpstrideConv2DFunctor<device::CPU,float> {
    void operator()(const Tensor<const float>& input,
                    const Tensor<const float>& kernel,
                    Tensor<float>& output);
};

}  // namespace upstride

#endif  //KERNEL_TIME_TWO_H_
