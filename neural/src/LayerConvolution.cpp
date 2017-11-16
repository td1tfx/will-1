#include "LayerConvolution.h"
#include "MatrixFiller.h"

LayerConvolution::LayerConvolution()
{
    //use_weight_ex = true;
}

LayerConvolution::~LayerConvolution()
{
    safe_delete({ &b_ex_, &as_b_ex_, &W_simple_, &dW_simple_, &as_W_simple_, &workspace_ });
    //safe_delete(as_x_pos);
}

void LayerConvolution::init2()
{
    out_channel_ = option_->getIntFromSection(layer_name_, "channel", 1);

    window_width_ = std::min(option_->getIntFromSection(layer_name_, "window_width", 1), prev_layer_->getOutWidth());
    window_height_ = std::min(option_->getIntFromSection(layer_name_, "window_height", window_width_), prev_layer_->getOutHeight());

    stride_width_ = option_->getIntFromSection(layer_name_, "stride_width", 1);
    stride_height_ = option_->getIntFromSection(layer_name_, "stride_height", stride_width_);
    padding_width_ = option_->getIntFromSection(layer_name_, "padding_width", 0);
    padding_height_ = option_->getIntFromSection(layer_name_, "padding_height", padding_width_);

    use_simple_weight_ = option_->getIntFromSection(layer_name_, "use_simple_weight", 0) != 0;

    out_width_ = ceil(1.0 * (prev_layer_->getOutWidth() + 2 * padding_width_ - window_width_) / stride_width_ + 1);
    out_height_ = ceil(1.0 * (prev_layer_->getOutHeight() + 2 * padding_height_ - window_height_) / stride_height_ + 1);
    out_total_ = out_width_ * out_height_ * out_channel_;
    X_ = new Matrix(out_width_, out_height_, out_channel_, batch_);
    dX_ = new Matrix(X_);
    dX_->initData(0);

    //卷积核矩阵
    //先创建多维卷积核，再决定是否使用
    W_ = new Matrix(window_width_, window_height_, prev_layer_->getOutChannel(), out_channel_);
    dW_ = new Matrix(W_);
    dW_->initData(0);

    int in_pamameter = prev_layer_->getOutChannel() * window_width_ * window_height_;
    int out_parameter = out_channel_ * window_width_ * window_height_;
    auto fill_type = option_->getRandomFillTypeFromSection(layer_name_, "weight_fill", "xavier");
    MatrixFiller::fill(W_, fill_type, in_pamameter, out_parameter);

    if (!use_simple_weight_)
    {
        //W_simple_ = W_->cloneShared();
        //dW_simple_ = dW_->cloneShared();
    }
    else
    {
        //可以使用1维的核进行复杂连接
        W_simple_ = new Matrix(window_width_, window_height_, 1, out_channel_);
        MatrixFiller::fill(W_, fill_type, in_pamameter, out_parameter);
        dW_simple_ = new Matrix(W_);
        //创建卷积核展开矩阵，W -> WW0W...取决于连接方式
        int w_size = window_width_ * window_height_ *  out_channel_;
        as_W_simple_ = new Matrix(w_size, w_size *  prev_layer_->getOutChannel(), MATRIX_DATA_INSIDE, CUDA_CPU);
        as_W_simple_->initData(0);

        //将连接信息读出来保存到connectRecord
        std::vector<std::vector<real>> connect_info;
        for (int i = 0; i < out_channel_; i++)
        {
            std::string str = convert::formatString("channel%d", i);
            str = option_->getStringFromSection(layer_name_, str, "");
            std::vector<real> d;
            convert::findNumbers(str, &d);
            connect_info.push_back(d);
        }

        for (int w = 0; w < window_width_; w++)
        {
            for (int h = 0; h < window_height_; h++)
            {
                for (int cp = 0; cp < prev_layer_->getOutChannel(); cp++)
                {
                    for (int c = 0; c < out_channel_; c++)
                    {
                        real v = 1;  //默认是都连的！
                        if (c < connect_info.size() && cp < connect_info[c].size())
                        {
                            v = connect_info[c][cp];
                        }
                        as_W_simple_->getData(W_simple_->whcn2i(w, h, 0, c), W_->whcn2i(w, h, cp, c)) = v;
                    }
                }
            }
        }
        as_W_simple_->toGPU();
        //依据connectMatrix将Weight展开为WeightEx，将bias展开为biasMatrix
        Matrix::mulVector(as_W_simple_, W_simple_, W_, 1, 0, MATRIX_TRANS);
    }

    //构造bias的展开矩阵
    if (need_bias_)
    {
        b_ex_ = new Matrix(X_);
        db_ = new Matrix(1, 1, out_channel_, 1);
        db_->initData(0);
        b_ = new Matrix(1, 1, out_channel_, 1);
        b_->initData(0);
        as_b_ex_ = new Matrix(out_channel_, out_width_ * out_height_ * out_channel_, MATRIX_DATA_INSIDE, CUDA_CPU);
        as_b_ex_->initData(0);

        for (int w = 0; w < out_width_; w++)
        {
            for (int h = 0; h < out_height_; h++)
            {
                for (int c = 0; c < out_channel_; c++)
                {
                    as_b_ex_->getData(c, w + out_width_ * h + out_width_ * out_height_ * c) = 1;
                }
            }
        }
        as_b_ex_->toGPU();
    }
    workspace_ = new Matrix(1, int(1e6));
}

void LayerConvolution::resetGroupCount2()
{
    b_ex_->resize(batch_);
}

void LayerConvolution::updateX()
{
    MatrixExtend::convolutionForward(prev_layer_->getA(), X_, W_, workspace_, workspace_forward_,
        stride_width_, stride_height_, padding_width_, padding_height_);
    if (need_bias_)
    {
        //若显存紧张，可以不开辟biasMatrix的空间，先处理到A矩阵
        b_ex_->initData(0);
        Matrix::mulVector(as_b_ex_, b_, b_ex_, 1, 0, MATRIX_TRANS);
        b_ex_->repeat();
        Matrix::add(X_, b_ex_, X_, 1, 1);
    }
    //BiasExMatrix->print();
    //LOG("ccc %g\n", X_->dotSelf());
}

//这里没有连dW和dB一起算出来
//如果严格遵守顺序应该是在updateParameters算出来
void LayerConvolution::updatePrevLayerDA()
{
    MatrixExtend::convolutionBackward(X_, dX_, prev_layer_->getA(), prev_layer_->getDA(), W_, nullptr, nullptr,
        workspace_, workspace_backward_dX_, workspace_backward_dW_,
        stride_width_, stride_height_, padding_width_, padding_height_, 1, 1);
}

//更新dweight
void LayerConvolution::updateDParameters(real momentum)
{
    MatrixExtend::convolutionBackward(X_, dX_, prev_layer_->getA(), nullptr, W_, dW_, db_,
        workspace_, workspace_backward_dX_, workspace_backward_dW_,
        stride_width_, stride_height_, padding_width_, padding_height_, 1, momentum);
    if (use_simple_weight_)
    {
        //dW-dW_simple_
        Matrix::mulVector(as_W_simple_, dW_, dW_simple_, 1.0 / prev_layer_->getOutChannel(), 0, MATRIX_NO_TRANS);
        //注意这里可能不正确，基类只更新W
        Matrix::mulVector(as_W_simple_, dW_simple_, dW_, 1, 0, MATRIX_TRANS);
    }
}

