#include "utils.hpp"

#include <algorithm>
#include <stdexcept>

#include "upstride.hpp"

using namespace upstride;

Padding upstride::paddingFromString(std::string paddingString) {
    if (paddingString == "SAME")
        return Padding::SAME;
    if (paddingString == "VALID")
        return Padding::VALID;
    if (paddingString == "EXPLICIT")
        return Padding::EXPLICIT;
    throw std::invalid_argument("Invalid padding encountered: " + paddingString);
}

DataFormat upstride::dataFormatFromString(std::string dataFormatString) {
    if (dataFormatString == "NHWC")
        return DataFormat::NHWC;
    if (dataFormatString == "NCHW")
        return DataFormat::NCHW;
    throw std::invalid_argument("Invalid data format encountered: " + dataFormatString);
}

int computeWindowedOutputSizeAndPadding(int inputSize, int filterSize,
                                        int dilation, int stride,
                                        Padding padding,
                                        int& paddingBefore,
                                        int& paddingAfter) {
    // Based on Tensorflow implementation:
    // https://github.com/tensorflow/tensorflow/blob/8f7e34982dde766b3fc73c90bcdbfccc001fe8e3/tensorflow/core/framework/kernel_shape_util.cc#L18-L65

    int effectiveFilterSize = (filterSize - 1) * dilation + 1;
    int outputSize;
    switch (padding) {
        case Padding::VALID:
            outputSize = (inputSize - effectiveFilterSize + stride) / stride;
            paddingBefore = paddingAfter = 0;
            break;
        case Padding::EXPLICIT:
            outputSize = (inputSize + paddingBefore + paddingAfter - effectiveFilterSize + stride) / stride;
            break;
        case Padding::SAME:
            outputSize = (inputSize + stride - 1) / stride;
            const int paddingNeeded = std::max(0, (outputSize - 1) * stride + effectiveFilterSize - inputSize);
            // For odd values of total padding, add more padding at the 'right' side of the given dimension.
            paddingBefore = paddingNeeded / 2;
            paddingAfter = paddingNeeded - paddingBefore;
            break;
    }

    return outputSize;
}

Shape upstride::computeConvOutputSize(const int typeDim, const DataFormat dataFormat, const Shape& inputShape, const Shape& filterShape,
                                      Padding paddingPreset, const std::vector<int32_t>& explicitPadding, const std::vector<int32_t>& stride, const std::vector<int32_t>& dilation) {
    // Assumptions on the filter dimensions are as follows:
    const int filterWidthDim = 0;
    const int filterHeightDim = 1;
    const int filterInChannelDim = 3;
    const int filterOutChannelDim = 4;

    // Perform shape checks
    if (inputShape.getSize() != 4)
        throw std::invalid_argument("Four-dimensional input tensor expected");
    if (filterShape.getSize() != 5)
        throw std::invalid_argument("Five-dimensional filter tensor expected");
    if (filterShape[0] != typeDim)
        throw std::invalid_argument("First filter dimension mismatch, got " + std::to_string(filterShape[0]));
    if (inputShape.depth(dataFormat) % filterShape[filterInChannelDim] != 0)
        throw std::invalid_argument("Filter channels number/input channels number mismatch");

    // Setting the last output dimension
    Shape outputShape(4);
    outputShape.depth(dataFormat) = filterShape[filterOutChannelDim];

    // init padding
    int padLeft, padRight, padTop, padBottom;
    if (paddingPreset == Padding::EXPLICIT) {
        padLeft = padRight = explicitPadding[0];
        padTop = padBottom = explicitPadding[1];
    }

    // compute output size
    //fixme: dilation is not taken into account properly
    outputShape.width(dataFormat) = computeWindowedOutputSizeAndPadding(
        inputShape.width(dataFormat), filterShape[filterWidthDim],
        dilation[0], stride[0], paddingPreset,
        padLeft, padRight);

    outputShape.height(dataFormat) = computeWindowedOutputSizeAndPadding(
        inputShape.height(dataFormat), filterShape[filterHeightDim],
        dilation[1], stride[1], paddingPreset,
        padTop, padBottom);

    return outputShape;
}
