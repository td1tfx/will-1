#pragma once
#include "Layer.h"

class LayerConvolution : public Layer
{
public:
    LayerConvolution();
    virtual ~LayerConvolution();

    int window_width_, window_height_;
    int stride_width_ = 1, stride_height_ = 1;  //步长
    int padding_width_ = 0, padding_height_ = 0;  //填充

    bool use_simple_weight_ = true;

    //方便计算的偏移矩阵，这个要考虑一下内存占用，可能太大了
    Matrix* b_ex_ = nullptr;
    Matrix* as_b_ex_ = nullptr;
    Matrix* workspace_ = nullptr;
    //辅助计算的数组，GPU模式下用来保存计算最快的方法
    std::vector<int> workspace_forward_, workspace_backward_dX_, workspace_backward_dW_;

private:
    //小卷积核，根据连接值展开，N，C均有对应值
    Matrix* W_simple_ = nullptr;

    //小卷积核变化
    Matrix* dW_simple_ = nullptr;

    Matrix* as_W_simple_ = nullptr;
    int as_inited_ = 0;

    //需要一个连接方式矩阵，看起来很麻烦
    //应该是从卷积核和计算方式算出一个矩阵，这个矩阵应该是比较稀疏的
    //提供的是连接方式，卷积核，据此计算出一个大矩阵
protected:
    void init2() override;
    void virtual resetGroupCount2() override;
    void updateX() override;
    void updatePrevLayerDA() override;
    void updateDParameters(real momentum) override;
};

