#pragma once
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace upstride {

enum class Padding {
    SAME,
    VALID,
    EXPLICIT
};

enum class DataFormat {
    NCHW,  // channel-first
    NHWC   // channel-last
};

inline int getWidthDimensionNumber(const DataFormat& dataFormat) {
    static const int DIM_NUMBERS[] = {3, 2};  // Dimension numbers matching DataFormat enumeration
    return DIM_NUMBERS[static_cast<int>(dataFormat)];
}

inline int getHeightDimensionNumber(const DataFormat& dataFormat) {
    static const int DIM_NUMBERS[] = {2, 1};  // Dimension numbers matching DataFormat enumeration
    return DIM_NUMBERS[static_cast<int>(dataFormat)];
}

inline int getDepthDimensionNumber(const DataFormat& dataFormat) {
    static int DIM_NUMBERS[] = {1, 3};  // Dimension numbers matching DataFormat enumeration
    return DIM_NUMBERS[static_cast<int>(dataFormat)];
}

/**
 * @brief A fixed size container of a numeric POD type
 * 
 * @tparam T    The datatype
 * @tparam N    The tuple size
 */
template <typename T, const int N>
class Tuple {
    T values[N];

   public:
    Tuple() {
        for (int i = 0; i < N; ++i)
            values[i] = 0;
    }

    T operator[](int i) const { return values[i]; }
    T& operator[](int i) { return values[i]; }

    operator T*() { return values; }
};

using IntPair = Tuple<int, 2>;

/**
 * @brief Represents shapes of a tensor
 * 
 */
class Shape {
    uint8_t size;
    int* shape;

   public:
    /**
    * @brief Construct a new Shape object
    * 
    * @param s Size of the shape array
    * @param _shape Array of shapes
    */
    Shape(int s, const int* _shape) : size(s) {
        shape = new int[s];
        for (int i = 0; i < s; i++) {
            shape[i] = _shape[i];
        }
    }
    /**
     * @brief Construct a new Shape object that creates a s size shape with all dimension to 0
     * 
     * @param s 
     */
    Shape(int s) : size(s) {
        shape = new int[s];
        for (int i = 0; i < s; i++) {
            shape[i] = 0;
        }
    }

    /**
     * @brief Destroy the Shape object
     * 
     */
    ~Shape() { delete[] shape; }
    uint8_t getSize() const { return size; }
    const int* getShapePtr() const { return shape; }

    /**
     * @brief Accesses shape dimension size by dimension index
     * 
     * @param i     A dimension index
     * @return the size of a corresponding dimension
     */
    int operator[](int i) const { return shape[i]; }
    int& operator[](int i) { return shape[i]; }

    /**
     * @brief Accesses the width dimension in function of a specific data format
     * 
     * @param fmt   The data format of the tensor
     * @return the tensor width.
     */
    int& width(const DataFormat& fmt) {
        return shape[getWidthDimensionNumber(fmt)];
    }
    int width(const DataFormat& fmt) const {
        return shape[getWidthDimensionNumber(fmt)];
    }

    /**
     * @brief Accesses the height dimension in function of a specific data format
     * 
     * @param fmt   The data format of the tensor
     * @return the tensor height.
     */
    int& height(const DataFormat& fmt) {
        return shape[getHeightDimensionNumber(fmt)];
    }
    int height(const DataFormat& fmt) const {
        return shape[getHeightDimensionNumber(fmt)];
    }

    /**
     * @brief Acessess the depth (channel) dimension in function of a specific data format
     * 
     * @param fmt   The data format of the tensor
     * @return the tensor depth.
     */
    int& depth(const DataFormat& fmt) {
        return shape[getDepthDimensionNumber(fmt)];
    }
    int depth(const DataFormat& fmt) const {
        return shape[getDepthDimensionNumber(fmt)];
    }
};

/**
 * @brief Tensor representation using its shape a tensor array
 * 
 * @tparam T Type of the tensor
 */
template <typename T>
class Tensor {
    const Shape shape;
    T* tensor;

   public:
    /**
    * @brief Construct a new Tensor object
    * 
    * @param sh Shape of the tensor
    * @param t Tensor
    */
    Tensor(const Shape& sh, T* t) : shape(sh.getSize(), sh.getShapePtr()),
                                    tensor(t) {}
    /**
     * @brief Get the pointer to the Tensor object 
     * 
     * @return T* Pointer to tensor
     */
    T* getTensorPtr() { return tensor; }

    /**
     * @brief Get the Shape object
     * 
     * @return const Shape& 
     */
    const Shape& getShape() const { return shape; }
};


/**
 * @brief Retrieves padding preset value from a string.
 * Raises an exception if unable to interpret the string.
 * @param paddingString     The string
 * @return corresponding padding value.
 */
Padding paddingFromString(std::string paddingString) {
    if (paddingString == "same")
        return Padding::SAME;
    if (paddingString == "valid")
        return Padding::VALID;
    if (paddingString == "explicit")
        return Padding::EXPLICIT;
    throw std::invalid_argument("Invalid padding encountered: " + paddingString);
}

/**
 * @brief Retrieves data format value from a string.
 * Raises an exception if unable to interpret the string.
 * @param dataFormatString     The string
 * @return corresponding data format value.
 */
DataFormat dataFormatFromString(std::string dataFormatString) {
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


/**
 * @brief Computes convolution output shape
 * The filter memory layout is assumed as follows: [blade, filter height, filter_width, input channels, output channels]
 * 
 * @param typeDim           Dimensionality of a specific UpStride datatype (e.g., 4 for quaternions)
 * @param dataFormat        Input and output tensors data format
 * @param inputShape        Input tensor shape
 * @param filterShape       Kernel tensor shape
 * @param paddingPreset     Padding preset
 * @param padding           Explicit padding value if the padding preset is explicit
 * @param stride            Convolution stride
 * @param dilation          Convolution dilation
 * @return the output tensor shape.
 */
Shape computeConvOutputSize(const int typeDim, const DataFormat dataFormat, const Shape& inputShape, const Shape& filterShape,
                            Padding paddingPreset, const IntPair& padding, const IntPair& stride, const IntPair& dilation);
                            Shape computeConvOutputSize(const int typeDim, const DataFormat dataFormat, const Shape& inputShape, const Shape& filterShape,
                            Padding paddingPreset, const IntPair& explicitPadding, const IntPair& stride, const IntPair& dilation) {
    // Assumptions on the filter dimensions are as follows:
    const int filterWidthDim = 0;
    const int filterHeightDim = 1;
    const int filterInChannelDim = 3;
    const int filterOutChannelDim = 4;

    // Perform shape checks
    if (inputShape.getSize() != 4)
        throw std::invalid_argument("Four-dimensional input expected");
    if (filterShape.getSize() != 5)
        throw std::invalid_argument("Five-dimensional input expected");
    if (filterShape[0] == typeDim)
        throw std::invalid_argument("First filter dimension mismatch");
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

}  // namespace upstride