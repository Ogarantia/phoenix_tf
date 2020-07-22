#include "tensorflow_includes.hpp"
#include "utils.hpp"


REGISTER_OP("UpstrideConv2D")
    .Attr("T: {int32, float}")
    .Input("input: T")
    .Input("kernel: T")
    .Output("output: T")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr(::tensorflow::GetPaddingAttrStringWithExplicit())
    .Attr(::tensorflow::GetExplicitPaddingsAttrString())
    .Attr(::tensorflow::GetConvnetDataFormatAttrString())
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      
      return ::tensorflow::errors::Unimplemented("");
    });