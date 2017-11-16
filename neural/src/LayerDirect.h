#pragma once
#include "Layer.h"

//直连层，含reshape的作用，数据的总输出与上一层完全一样，默认情况维度也一样，一些特殊情况有需要，例如两次激活等
//反向有效率问题，尽量少用
class LayerDirect : public Layer
{
public:
    LayerDirect();
    virtual ~LayerDirect();

protected:
    void init2() override;
    void updateX() override { MatrixExtend::activeForward(ACTIVE_FUNCTION_NONE, prev_layer_->getA(), X_); }
    void updatePrevLayerDA() override { Matrix::add(dX_, prev_layer_->getDA(), prev_layer_->getDA()); }
};

