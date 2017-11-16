#pragma once
#include "Layer.h"

//神经层管理
//一般来说不要直接使用Layer的子类，而是创建基类的指针
class LayerCreator : Neural
{
private:
    LayerCreator() {}
    ~LayerCreator() {}
public:
    static Layer* createByConnectionType(LayerConnectionType connection_type);
};
