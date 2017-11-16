#include "LayerNone.h"

LayerNone::LayerNone()
{
}


LayerNone::~LayerNone()
{
}

//这里的特别设置部分实际对应输入层，即空连接层
void LayerNone::init2()
{
    //输入层设置A
    if (visible_type_ == LAYER_VISIBLE_IN)
    {
        //图片模式优先
        out_width_ = option_->getIntFromSection(layer_name_, "width", 0);
        out_height_ = option_->getIntFromSection(layer_name_, "height", 0);
        out_channel_ = option_->getIntFromSection(layer_name_, "channel", 1);

        if (out_width_ <= 0 || out_height_ <= 0 || out_channel_ <= 0)
        {
            out_width_ = 1;
            out_height_ = 1;
            out_channel_ = option_->getIntFromSection(layer_name_, "node", 0);
        }

        out_total_ = out_width_ * out_height_ * out_channel_;

        X_ = new Matrix(out_width_, out_height_, out_channel_, batch_, MATRIX_DATA_OUTSIDE);
        dX_ = new Matrix(X_);
    }
}