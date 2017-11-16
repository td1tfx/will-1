#pragma once
#include "Layer.h"

class LayerNone : public Layer
{
public:
    LayerNone();
    virtual ~LayerNone();
protected:
    void init2();
};

