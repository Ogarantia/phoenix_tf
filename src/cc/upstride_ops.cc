#include "tensorflow_includes.hpp"


REGISTER_OP("UpstrideConv2D")
    .Attr("T: {int32, float}")
    .Input("input: T")
    .Input("kernel: T")
    .Output("output: T")
    .SetShapeFn(tensorflow::shape_inference::Conv2DShapeWithExplicitPadding);