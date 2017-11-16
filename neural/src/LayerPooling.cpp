#include "LayerPooling.h"
#include "Net.h"

LayerPooling::LayerPooling()
{
}

LayerPooling::~LayerPooling()
{
    //safe_delete(record_pos);
}

//采样层，参数为本层横向和纵向的采样像素个数
void LayerPooling::init2()
{
    out_channel_ = prev_layer_->getOutChannel();

    reverse_ = option_->getIntFromSection(layer_name_, "reverse", 0);

    window_width_ = std::min(option_->getIntFromSection(layer_name_, "window_width", 1), prev_layer_->getOutWidth());
    window_height_ = std::min(option_->getIntFromSection(layer_name_, "window_height", window_width_), prev_layer_->getOutHeight());

    stride_width_ = option_->getIntFromSection(layer_name_, "stride_width", window_width_);
    stride_height_ = option_->getIntFromSection(layer_name_, "stride_height", window_height_);
    padding_width_ = option_->getIntFromSection(layer_name_, "padding_width", 0);
    padding_height_ = option_->getIntFromSection(layer_name_, "padding_height", 0);

    //网络设计者应注意，此处如不能整除，结果通常不正常
    if (reverse_ == 0)
    {
        out_width_ = ceil(1.0 * (prev_layer_->getOutWidth() + 2 * padding_width_ - window_width_) / stride_width_ + 1);
        out_height_ = ceil(1.0 * (prev_layer_->getOutHeight() + 2 * padding_height_ - window_height_) / stride_height_ + 1);
    }
    else
    {
        out_width_ = stride_width_ * (prev_layer_->getOutWidth() - 1) + window_width_ - 2 * padding_width_;
        out_height_ = stride_height_ * (prev_layer_->getOutHeight() - 1) + window_height_ - 2 * padding_height_;
    }
    out_total_ = out_channel_ * out_height_ * out_width_;

    X_ = new Matrix(out_width_, out_height_, out_channel_, batch_);
    dX_ = new Matrix(X_);

    pooling_type_ = option_->getPoolingTypeFromSection(layer_name_, "pool_type", reverse_ == 0 ? "max" : "average");

    nearest_ = option_->getIntFromSection(layer_name_, "nearest", 1);

    layer_ref_ = this;
    //如果是最大值反池化，则需要一个参考的pooling层
    if (reverse_ && pooling_type_ == POOLING_MAX)
    {
        layer_ref_ = dynamic_cast<Net*>(net_)->getLayerByName(option_->getStringFromSection(layer_name_, "pooling_ref"));
    }
}

void LayerPooling::updateX()
{
    if (reverse_ == 0)
    {
        MatrixExtend::poolingForward(prev_layer_->getA(), X_, pooling_type_, window_width_, window_height_, stride_width_, stride_height_, padding_width_, padding_height_);
    }
    else
    {
        //这里利用池化的逆操作
        if (pooling_type_ != POOLING_MAX)
        {
            MatrixExtend::poolingBackward(prev_layer_->getA(), prev_layer_->getA(), X_, X_, pooling_type_,
                window_width_, window_height_, stride_width_, stride_height_, padding_width_, padding_height_);
            if (nearest_) { X_->scale(window_width_ * window_height_); }
        }
        else
        {
            MatrixExtend::poolingBackward(layer_ref_->getPrevLayer()->getA(), prev_layer_->getA(), layer_ref_->getX(), X_, pooling_type_,
                window_width_, window_height_, stride_width_, stride_height_, padding_width_, padding_height_);
        }
    }
}

void LayerPooling::updatePrevLayerDA()
{
    if (reverse_ == 0)
    {
        MatrixExtend::poolingBackward(X_, dX_, prev_layer_->getA(), prev_layer_->getDA(), pooling_type_,
            window_width_, window_height_, stride_width_, stride_height_, padding_width_, padding_height_, 1, 1);
    }
    else
    {
        if (pooling_type_ != POOLING_MAX)
        {
            MatrixExtend::poolingForward(dX_, prev_layer_->getDA(), pooling_type_,
                window_width_, window_height_, stride_width_, stride_height_, padding_width_, padding_height_, 1, 1);
            if (nearest_) { prev_layer_->getDA()->scale(1.0 / (window_width_ * window_height_)); }
        }
        else
        {
            //max模式反向未完成，直觉上似乎毫无意义
        }
    }
}

