#pragma once
#include "Layer.h"

class LayerFullConnect : public Layer
{
public:
    LayerFullConnect();
    virtual ~LayerFullConnect();

    //更新偏移向量的辅助向量，所有值为1，维度为数据组数
    Matrix* as_b_ = nullptr;

protected:
    void init2() override;
    void resetGroupCount2() override;
    void updateX() override;
    void updatePrevLayerDA() override;
    void updateDParameters(real momentum) override;
};

