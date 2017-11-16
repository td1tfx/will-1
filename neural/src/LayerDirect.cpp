#include "LayerDirect.h"

LayerDirect::LayerDirect()
{
}


LayerDirect::~LayerDirect()
{
}

void LayerDirect::init2()
{
    out_total_ = prev_layer_->getOutTotal();

    out_width_ = option_->getIntFromSection(layer_name_, "width", 0);
    out_height_ = option_->getIntFromSection(layer_name_, "height", 0);
    out_channel_ = option_->getIntFromSection(layer_name_, "channel", 1);

    if (out_total_ != out_width_ * out_height_ * out_channel_)
    {
        out_width_ = prev_layer_->getOutWidth();
        out_height_ = prev_layer_->getOutHeight();
        out_channel_ = prev_layer_->getOutChannel();
    }

    X_ = prev_layer_->getA()->cloneShared();
    dX_ = new Matrix(X_);
}

