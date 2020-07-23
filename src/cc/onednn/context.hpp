#pragma once
#include "dnnl.hpp"
#include "upstride.hpp"

namespace upstride {
namespace onednn {

/**
 * @brief Converts a common data type to oneDNN data type handle
 * @tparam T    The input type
 * @return oneDNN data type
 */
template <typename T>
dnnl::memory::data_type getDataType();
template <>
dnnl::memory::data_type getDataType<float>() { return dnnl::memory::data_type::f32; }

/**
 * @brief Retrieves oneDNN memory format tag corresponding to a given data format.
 * @param df the data format.
 * @return dnnl::memory::format_tag 
 */
dnnl::memory::format_tag convertDataFormatToFormatTag(DataFormat df) {
    switch (df) {
        case DataFormat::NCHW:
            return dnnl::memory::format_tag::nchw;
        case DataFormat::NHWC:
            return dnnl::memory::format_tag::nhwc;
        default:
            throw std::invalid_argument("Unimplemented valid DataFormat.");
    }
}

class Context : public upstride::Context {
    dnnl::engine oneEngine;
    dnnl::stream oneStream;

   public:
    Context(const int typeDim) : upstride::Context(typeDim), oneEngine(dnnl::engine::kind::cpu, 0), oneStream(oneEngine) {}

    /**
     * @brief Retrieves oneDNN engine instance associated with the current context.
     * @return a reference to a dnnl::engine object.
     */
    const dnnl::engine& getEngine() const { return oneEngine; }
    dnnl::engine& getEngine() { return oneEngine; }
};

/**
 * @brief Client memory to oneDNN memory wrapping
 */
class Memory {
    dnnl::memory::desc md;
    dnnl::memory mem;

   public:
    template <typename T>
    Memory(const Tensor<const T>& tensor, DataFormat df, const Context& context) : md(dnnl::memory::dims(tensor.getShape().getShapePtr(), tensor.getShape().getShapePtr() + tensor.getShape().getSize()),
                                                                                      getDataType<T>(),
                                                                                      convertDataFormatToFormatTag(df)),
                                                                                   mem(md, context.getEngine(), const_cast<float*>(tensor.getDataPtr())) {
    }

    template <typename T>
    Memory(Tensor<T>& tensor, DataFormat df, const Context& context) : md(dnnl::memory::dims(tensor.getShape().getShapePtr(), tensor.getShape().getShapePtr() + tensor.getShape().getSize()),
                                                                          getDataType<T>(),
                                                                          convertDataFormatToFormatTag(df)),
                                                                       mem(md, context.getEngine(), tensor.getDataPtr()) {
    }
};

}  // namespace onednn
}  // namespace upstride
