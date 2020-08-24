#include "tensorflow_includes.hpp"
#include "upstride.hpp"
#include "upstride_tf.hpp"

#define GET_FIELD(CTX, NAME, VAR)                                                                           \
    {                                                                                                       \
        if (!CTX->GetAttr(NAME, &VAR).ok()) return tensorflow::errors::InvalidArgument("Cannot get " NAME); \
    }

REGISTER_OP("UpstrideConv2D")
    .Attr("T: {int32, float}")
    .Attr("uptype: int = 0")
    .Input("input: T")
    .Input("kernel: T")
    .Output("output: T")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
#ifdef TENSORFLOW_VERSION_1
    .Attr(tensorflow::GetPaddingAttrString())
#else
    .Attr(tensorflow::GetPaddingAttrStringWithExplicit())
    .Attr(tensorflow::GetExplicitPaddingsAttrString())
#endif
    .Attr(tensorflow::GetConvnetDataFormatAttrString())
    .Attr("groups: int = 1")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* ctx) {
        static const int
            INPUT_IDX = 0,
            FILTER_IDX = 1;

        try {
            // get the algebra
            int uptype;
            GET_FIELD(ctx, "uptype", uptype);
            const upstride::Algebra algebra = upstride::getAlgebraFromType(uptype);

            // get data format
            std::string dataFormatStr;
            GET_FIELD(ctx, "data_format", dataFormatStr);
            const upstride::DataFormat dataFormat = upstride::dataFormatFromString(dataFormatStr);

            // get padding
            std::string paddingStr;
            GET_FIELD(ctx, "padding", paddingStr);
            const upstride::Padding padding = upstride::paddingFromString(paddingStr);
            upstride::IntTuple explicitPadding;
            upstride::IntPair padBefore, padAfter;
            if (padding == upstride::Padding::EXPLICIT) {
                GET_FIELD(ctx, "explicit_paddings", explicitPadding);
                // fixme: this is pretty much not how explicit padding must be implemented
                if (!getSpatialStep(explicitPadding, 1, padBefore))
                    return tensorflow::errors::InvalidArgument("Invalid explicit paddings");
                padAfter = padBefore;
            }

            // get stride and dilations
            upstride::IntPair stride, dilation;
            upstride::IntTuple tuple;
            GET_FIELD(ctx, "strides", tuple);
            if (!getSpatialStep(tuple, 1, stride))
                return tensorflow::errors::InvalidArgument("Invalid stides");
            GET_FIELD(ctx, "dilations", tuple);
            if (!getSpatialStep(tuple, 1, dilation))
                return tensorflow::errors::InvalidArgument("Invalid dilations");

            // compute output shape assuming OIHW filter layout
            using namespace upstride::frontend_tf;
            tensorflow::shape_inference::DimensionHandle outWidth, outHeight;
            auto inputShape = ctx->input(INPUT_IDX);
            auto filterShape = ctx->input(FILTER_IDX);
            const int filterDims = ctx->Rank(filterShape);

            auto result = upstride::frontend_tf::computeWindowedOutputSize(ctx,
                ctx->Dim(inputShape, getWidthDimensionNumber(dataFormat)),
                ctx->Dim(filterShape, filterDims - 1),
                dilation.x, stride.x, padding, padBefore.x, padAfter.x, outWidth);
            if (!result.ok())
                return result;

            result = upstride::frontend_tf::computeWindowedOutputSize(ctx,
                ctx->Dim(inputShape, getHeightDimensionNumber(dataFormat)),
                ctx->Dim(filterShape, filterDims - 2),
                dilation.y, stride.y, padding, padBefore.y, padAfter.y, outHeight);
            if (!result.ok())
                return result;

            switch (dataFormat) {
                case upstride::DataFormat::NCHW:
                    ctx->set_output(0, ctx->MakeShape({
                        ctx->Dim(inputShape, 0), ctx->Dim(filterShape, 0), outHeight, outWidth
                    }));
                    break;

                case upstride::DataFormat::NHWC:
                    ctx->set_output(0, ctx->MakeShape({
                        ctx->Dim(inputShape, 0), outHeight, outWidth, ctx->Dim(filterShape, 0)
                    }));
                    break;

                default:
                    return tensorflow::errors::InvalidArgument("Unhandled data format: %s", dataFormatStr);
            }
        }

        catch (std::invalid_argument& ex) {
            return tensorflow::errors::InvalidArgument(ex.what());
        }

        return tensorflow::Status::OK();
    });

REGISTER_OP("UpstrideConv2DGrad")
    .Attr("T: {int32, float}")
    .Attr("uptype: int = 0")
    .Input("grad: T")
    .Input("input: T")
    .Input("kernel: T")
    .Output("input_grad: T")
    .Output("kernel_grad: T")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
#ifdef TENSORFLOW_VERSION_1
    .Attr(tensorflow::GetPaddingAttrString())
#else
    .Attr(tensorflow::GetPaddingAttrStringWithExplicit())
    .Attr(tensorflow::GetExplicitPaddingsAttrString())
#endif
    .Attr(tensorflow::GetConvnetDataFormatAttrString())
    .Attr("groups: int = 1")
    .Attr("require_input_grad: bool = true")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        return tensorflow::Status::OK();
    });
