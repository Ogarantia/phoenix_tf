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
    .Attr("groups: int = 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return ::tensorflow::errors::Unimplemented("");
    });

REGISTER_OP("UpstrideConv2DGrad")
    .Attr("T: {int32, float}")
    .Input("grad: T")
    .Input("input: T")
    .Input("kernel: T")
    .Output("input_grad: T")
    .Output("kernel_grad: T")
    .Attr("require_input_grad: bool = true")
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
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        return ::tensorflow::Status::OK();
    });
