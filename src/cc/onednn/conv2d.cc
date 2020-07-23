
#include <iostream>

#include "dnnl.hpp"
#include "onednn/context.hpp"
#include "upstride.hpp"

static upstride::onednn::Context context(1);

namespace upstride {

void UpstrideConv2DFunctor<device::CPU, float>::operator()(
    const Tensor<const float>& inputTensor,
    const Tensor<const float>& kernelTensor,
    Tensor<float>& outputTensor,
    DataFormat df) {
    std::cout << "Coucou c'est oneDNN" << std::endl;


    upstride::onednn::Memory input(inputTensor, df, context);
    // upstride::onednn::Memory kernel(kernelTensor, df, context);
    upstride::onednn::Memory output(outputTensor, df, context);
    
    
}

}  // namespace upstride
