#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("UpstrideInput")
    .Attr("T: {int32, float}")
    .Input("input1: T")
    .Input("input2: T")
    .Input("input3: T")
    .Input("input4: T")
    .Output("input_processed1: T")
    .Output("input_processed2: T")
    .Output("input_processed3: T")
    .Output("input_processed4: T")
    .Output("input_processed5: T")
    .Output("input_processed6: T")
    .Output("input_processed7: T")
    .Output("input_processed8: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        c->set_output(2, c->input(0));
        c->set_output(3, c->input(0));
        c->set_output(4, c->input(0));
        c->set_output(5, c->input(0));
        c->set_output(6, c->input(0));
        c->set_output(7, c->input(0));
        return Status::OK();
    });

REGISTER_OP("UpstrideKernel")
    .Attr("T: {int32, float}")
    .Input("kernel1: T")
    .Input("kernel2: T")
    .Input("kernel3: T")
    .Input("kernel4: T")
    .Output("kernel_processed1: T")
    .Output("kernel_processed2: T")
    .Output("kernel_processed3: T")
    .Output("kernel_processed4: T")
    .Output("kernel_processed5: T")
    .Output("kernel_processed6: T")
    .Output("kernel_processed7: T")
    .Output("kernel_processed8: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        c->set_output(2, c->input(0));
        c->set_output(3, c->input(0));
        c->set_output(4, c->input(0));
        c->set_output(5, c->input(0));
        c->set_output(6, c->input(0));
        c->set_output(7, c->input(0));
        return Status::OK();
    });

REGISTER_OP("UpstrideOutput")
    .Attr("T: {int32, float}")
    .Input("output1: T")
    .Input("output2: T")
    .Input("output3: T")
    .Input("output4: T")
    .Input("output5: T")
    .Input("output6: T")
    .Input("output7: T")
    .Input("output8: T")
    .Output("output_processed1: T")
    .Output("output_processed2: T")
    .Output("output_processed3: T")
    .Output("output_processed4: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        c->set_output(2, c->input(0));
        c->set_output(3, c->input(0));
        return Status::OK();
    });