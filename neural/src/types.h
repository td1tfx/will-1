#pragma once

//包含一些类型定义

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"
#include "cudnn.h"
#include "will_cuda.h"

#if defined(__arm__) || defined(__aarch64__)
#else
#include "nvml.h"
#endif

//#define _NO_CUDA
#if CUDNN_VERSION < 5000 || defined(_NO_CUDA)
#include "cuda_hack.h"
#endif

#define VAR_NAME(a) #a

//默认使用单精度，整个工程除公用部分，均应当使用real定义浮点数
#ifndef _DOUBLE_PRECISION
#define _SINGLE_PRECISION
typedef float real;
#else
typedef double real;
#endif

#ifdef _SINGLE_PRECISION
#define REAL_MAX FLT_MAX
#define MYCUDNN_DATA_REAL CUDNN_DATA_FLOAT
#else
#define REAL_MAX DBL_MAX
#define MYCUDNN_DATA_REAL CUDNN_DATA_DOUBLE
#define curandGenerateUniform curandGenerateUniformDouble
#endif

#define WILL_NAMESPACE_BEGIN namespace will {
#define WILL_NAMESPACE_END };

//激活函数种类
typedef enum
{
    ACTIVE_FUNCTION_NONE = -1,
    ACTIVE_FUNCTION_SIGMOID = CUDNN_ACTIVATION_SIGMOID,
    ACTIVE_FUNCTION_RELU = CUDNN_ACTIVATION_RELU,
    ACTIVE_FUNCTION_TANH = CUDNN_ACTIVATION_TANH,
    ACTIVE_FUNCTION_CLIPPED_RELU = CUDNN_ACTIVATION_CLIPPED_RELU,
    ACTIVE_FUNCTION_ELU = CUDNN_ACTIVATION_ELU,   //note: it should be ELU in cudnn 6
    ACTIVE_FUNCTION_SCALE,
    ACTIVE_FUNCTION_SOFTMAX,
    ACTIVE_FUNCTION_SOFTMAX_FAST,
    ACTIVE_FUNCTION_SOFTMAX_LOG,
    ACTIVE_FUNCTION_FINDMAX,
    ACTIVE_FUNCTION_DROPOUT,
    ACTIVE_FUNCTION_RECURRENT,
    ACTIVE_FUNCTION_SOFTPLUS,    //only CPU
    ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION,  //only CUDA, same to below
    ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION,
    ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION,
    ACTIVE_FUNCTION_BATCH_NORMALIZATION,
    ACTIVE_FUNCTION_SPATIAL_TRANSFORMER,
    ACTIVE_FUNCTION_SQUARE,
} ActiveFunctionType;

typedef enum
{
    ACTIVE_PHASE_TRAIN,   //训练
    ACTIVE_PHASE_TEST,    //测试
} ActivePhaseType;

//池化种类，与cuDNN直接对应，可以类型转换
typedef enum
{
    POOLING_MAX = CUDNN_POOLING_MAX,
    POOLING_AVERAGE_PADDING = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    POOLING_AVERAGE_NOPADDING = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
} PoolingType;

//合并种类
typedef enum 
{
    COMBINE_CONCAT,
    COMBINE_ADD,
} CombineType;

//代价函数种类
typedef enum
{
    COST_FUNCTION_RMSE,
    COST_FUNCTION_CROSS_ENTROPY,
    COST_FUNCTION_LOG_LIKELIHOOD,
} CostFunctionType;

//for layer
//隐藏，输入，输出
typedef enum
{
    LAYER_VISIBLE_HIDDEN,
    LAYER_VISIBLE_IN,
    LAYER_VISIBLE_OUT,
} LayerVisibleType;

//连接类型
typedef enum
{
    LAYER_CONNECTION_NONE,                //无连接，用于输入层，不需要特殊设置
    LAYER_CONNECTION_FULLCONNECT,         //全连接
    LAYER_CONNECTION_CONVOLUTION,         //卷积
    LAYER_CONNECTION_POOLING,             //池化
    LAYER_CONNECTION_DIRECT,              //直连
    LAYER_CONNECTION_COMBINE,             //合并
} LayerConnectionType;

//for net

//初始化权重模式
typedef enum
{
    RANDOM_FILL_CONSTANT,
    RANDOM_FILL_XAVIER,
    RANDOM_FILL_GAUSSIAN,
    RANDOM_FILL_MSRA,
} RandomFillType;

//调整学习率模式
typedef enum
{
    ADJUST_LEARN_RATE_FIXED,
    ADJUST_LEARN_RATE_INV,
    ADJUST_LEARN_RATE_STEP,
} AdjustLearnRateType;

//
typedef enum
{
    BATCH_NORMALIZATION_PER_ACTIVATION = CUDNN_BATCHNORM_PER_ACTIVATION,
    BATCH_NORMALIZATION_SPATIAL = CUDNN_BATCHNORM_SPATIAL,
    BATCH_NORMALIZATION_AUTO,
} BatchNormalizationType;

typedef enum
{
    RECURRENT_RELU = CUDNN_RNN_RELU,
    RECURRENT_TANH = CUDNN_RNN_TANH,
    RECURRENT_LSTM = CUDNN_LSTM,
    RECURRENT_GRU = CUDNN_GRU,
} RecurrentType;

typedef enum
{
    RECURRENT_DIRECTION_UNI = CUDNN_UNIDIRECTIONAL,
    RECURRENT_DIRECTION_BI = CUDNN_BIDIRECTIONAL,
} RecurrentDirectionType;

typedef enum
{
    RECURRENT_INPUT_LINEAR = CUDNN_LINEAR_INPUT,
    RECURRENT_INPUT_SKIP = CUDNN_SKIP_INPUT,
} RecurrentInputType;


typedef enum
{
    RECURRENT_ALGO_STANDARD = CUDNN_RNN_ALGO_STANDARD,
    RECURRENT_ALGO_PERSIST_STATIC = CUDNN_RNN_ALGO_PERSIST_STATIC,
    RECURRENT_ALGO_PERSIST_DYNAMIC = CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
} RecurrentAlgoType;

typedef enum
{
    SOLVER_SGD,
    SOLVER_NAG,
    SOLVER_ADA_DELTA,
} SolverType;

typedef enum
{
    WORK_MODE_NORMAL,
    WORK_MODE_PRUNE,
    WORK_MODE_GAN,
} WorkModeType;

typedef enum
{
    PRUNE_ACTIVE,
    PRUNE_WEIGHT,
} PruneType;

