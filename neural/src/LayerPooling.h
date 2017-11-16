#pragma once
#include "Layer.h"
#include <vector>

class LayerPooling : public Layer
{
public:
    LayerPooling();
    virtual ~LayerPooling();

    PoolingType pooling_type_ = POOLING_MAX;
    //real Weight, Bias;
    //所有值为1
    //Matrix* _asBiasMatrix = nullptr;

    int window_width_, window_height_;  //pooling窗口尺寸
    int stride_width_, stride_height_;  //pooling步长
    int padding_width_, padding_height_;  //填充

    int reverse_ = 0;
    int nearest_ = 1;
    Layer* layer_ref_ = nullptr;  //用于参考的pooling层，max模式下有效

protected:
    void init2() override;
    void updateX() override;
    void updatePrevLayerDA() override;
};

