
#include <iostream>

#include "dnnl.hpp"
#include "upstride.hpp"

namespace upstride {

void UpstrideConv2DFunctor<device::CPU, float>::operator()(
    const Tensor<const float>& input,
    const Tensor<const float>& kernel,
    Tensor<float>& output) {
    std::cout << "Coucou c'est oneDNN" << std::endl;
}

}  // namespace upstride
