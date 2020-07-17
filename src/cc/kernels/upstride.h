#ifndef KERNEL_UPSTRIDE_OPS_H_
#define KERNEL_UPSTRIDE_OPS_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct UpstrideKernelFunctor {
    void operator()(const Device& d, int size, const T* kernel_1,
                    const T* kernel_2,
                    const T* kernel_3,
                    const T* kernel_4,
                    T* kernel_processed_1,
                    T* kernel_processed_2,
                    T* kernel_processed_3,
                    T* kernel_processed_4,
                    T* kernel_processed_5,
                    T* kernel_processed_6,
                    T* kernel_processed_7,
                    T* kernel_processed_8);
};

template <typename Device, typename T>
struct UpstrideInputFunctor {
    void operator()(const Device& d, int size, const T* input_1,
                    const T* input_2,
                    const T* input_3,
                    const T* input_4,
                    T* input_processed_1,
                    T* input_processed_2,
                    T* input_processed_3,
                    T* input_processed_4,
                    T* input_processed_5,
                    T* input_processed_6,
                    T* input_processed_7,
                    T* input_processed_8);
};

template <typename Device, typename T>
struct UpstrideOutputFunctor {
    void operator()(const Device& d, int size, const T* output_1,
                    const T* output_2,
                    const T* output_3,
                    const T* output_4,
                    const T* output_5,
                    const T* output_6,
                    const T* output_7,
                    const T* output_8,
                    T* output_processed_1,
                    T* output_processed_2,
                    T* output_processed_3,
                    T* output_processed_4);
};


template <typename Device, typename T>
struct UpstrideConv2DFunctor {
    /**
     * @brief 
     * 
     * @param inputShape 
     * @param kernelShape 
     * @param outputShape 
     * @param input 
     * @param kernel 
     * @param ouput 
     */
    void operator()(const Shape& inputShape,
                    const Shape& kernelShape,
                    const Shape& outputShape, 
                    const T* input, 
                    const T* kernel, 
                    T* ouput);
};

}  // namespace functor

}  // namespace tensorflow

#endif  //KERNEL_TIME_TWO_H_
