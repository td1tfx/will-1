#include "LayerFullConnect.h"
#include "MatrixFiller.h"
#include "Net.h"

LayerFullConnect::LayerFullConnect()
{

}


LayerFullConnect::~LayerFullConnect()
{
    safe_delete(as_b_);
}

//全连接层中，x1是本层输出数
void LayerFullConnect::init2()
{
    //可以用参考层来设置本层的尺寸
    auto layer_ref = dynamic_cast<Net*>(net_)->getLayerByName(option_->getStringFromSection(layer_name_, "reference"));
    if (layer_ref)
    {
        out_width_ = layer_ref->getOutWidth();
        out_height_ = layer_ref->getOutHeight();
        out_channel_ = layer_ref->getOutChannel();
    }
    else
    {
        out_width_ = option_->getIntFromSection(layer_name_, "width", 0);
        out_height_ = option_->getIntFromSection(layer_name_, "height", 0);
        out_channel_ = option_->getIntFromSection(layer_name_, "channel", 1);
    }
    if (out_width_ <= 0 || out_height_ <= 0 || out_channel_ <= 0)
    {
        out_width_ = 1;
        out_height_ = 1;
        out_channel_ = option_->getIntFromSection(layer_name_, "node", 0);
    }

    out_total_ = out_width_ * out_height_ * out_channel_;

    X_ = new Matrix(out_width_, out_height_, out_channel_, batch_);
    dX_ = new Matrix(X_);

    //weight矩阵，对于全连接层，行数是本层的节点数，列数是上一层的节点数
    W_ = new Matrix(out_total_, prev_layer_->getOutTotal());

    auto fill_type = option_->getRandomFillTypeFromSection(layer_name_, "weight_fill", "xavier");
    MatrixFiller::fill(W_, fill_type, prev_layer_->getOutTotal(), out_total_);

    dW_ = new Matrix(W_);
    dW_->initData(0);

    //偏移向量，维度为本层节点数
    if (need_bias_)
    {
        b_ = new Matrix(out_total_, 1);
        db_ = new Matrix(b_);
        b_->initData(0);
        db_->initData(0);

        as_b_ = new Matrix(batch_, 1);
        as_b_->initData(1);
        //output->print();
    }

    //一般这个只使用于隐藏的第一层，如果已知某些量特别重要，则初始化的时候特意增加该量对应的权重
    int weight_special = option_->getIntFromSection(layer_name_, "weight_special", 0);
    if (weight_special > 0)
    {
        real weight_special_zoom = option_->getRealFromSection(layer_name_, "weight_special_zoom", 1);
        W_->toCPU();
        for (int i = 0; i < this->out_total_; i++)
        {
            for (int j = 0; j < weight_special; j++)
            {
                W_->getData(i, j) *= weight_special_zoom;
            }
        }
        W_->toGPU();
    }
    prune_record_.resize(out_total_);
}

void LayerFullConnect::resetGroupCount2()
{
    if (as_b_->resize(batch_, 1) > 0)
    {
        as_b_->initData(1);
    }
}

void LayerFullConnect::updateX()
{
    if (need_bias_)
    {
        Matrix::copyData(b_, X_);
        X_->repeat();
        Matrix::mul(W_, prev_layer_->getA(), X_, 1, 1);
    }
    else
    {
        Matrix::mul(W_, prev_layer_->getA(), X_, 1, 0);
    }
    //LOG("%s, %g, %g\n", layer_name_.c_str(), X_->dotSelf(), prev_layer_->getA()->dotSelf());
}

void LayerFullConnect::updatePrevLayerDA()
{
    Matrix::mul(W_, dX_, prev_layer_->getDA(), 1, 1, MATRIX_TRANS, MATRIX_NO_TRANS);
}

void LayerFullConnect::updateDParameters(real momentum)
{
    Matrix::mul(dX_, prev_layer_->getA(), dW_, 1, momentum, MATRIX_NO_TRANS, MATRIX_TRANS);
    if (need_bias_)
    {
        Matrix::mulVector(dX_, as_b_, db_, 1, momentum, MATRIX_NO_TRANS);
    }
}

