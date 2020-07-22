#include "tensorflow_includes.hpp"
#include "utils.hpp"


REGISTER_OP("UpstrideConv2D")
    .Attr("T: {int32, float}")
    .Input("input: T")
    .Input("kernel: T")
    .Output("output: T")
    .Attr("strides: list(int)")
    .Attr(::tensorflow::GetPaddingAttrString())
    .Attr(::tensorflow::GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      
      return ::tensorflow::errors::Unimplemented("");
    });