#include "LayerCreator.h"
#include "LayerConvolution.h"
#include "LayerFullConnect.h"
#include "LayerPooling.h"
#include "LayerDirect.h"
#include "LayerNone.h"

Layer* LayerCreator::createByConnectionType(LayerConnectionType connection_type)
{
    Layer* layer = nullptr;
    switch (connection_type)
    {
    case LAYER_CONNECTION_FULLCONNECT:
        layer = new LayerFullConnect();
        break;
    case LAYER_CONNECTION_CONVOLUTION:
        layer = new LayerConvolution();
        break;
    case LAYER_CONNECTION_POOLING:
        layer = new LayerPooling();
        break;
    case LAYER_CONNECTION_DIRECT:
        layer = new LayerDirect();
        break;
    case LAYER_CONNECTION_NONE:
        layer = new LayerNone();
        break;
    default:
        layer = new Layer();
        break;
    }
    layer->setConnection(connection_type);
    return layer;
}

