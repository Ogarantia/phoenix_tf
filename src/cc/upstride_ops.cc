#include "tensorflow_includes.hpp"
#include "utils.hpp"


REGISTER_OP("UpstrideConv2D")
    .Attr("T: {int32, float}")
    .Input("input: T")
    .Input("kernel: T")
    .Output("output: T")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
#ifdef TENSORFLOW_VERSION_1
    .Attr(::tensorflow::GetPaddingAttrString())
#else
    .Attr(::tensorflow::GetPaddingAttrStringWithExplicit())
    .Attr(::tensorflow::GetExplicitPaddingsAttrString())
#endif
    .Attr(::tensorflow::GetConvnetDataFormatAttrString())
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      return ::tensorflow::errors::Unimplemented("");
    });