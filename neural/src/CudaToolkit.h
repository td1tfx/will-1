#pragma once
#include "types.h"
#include "cublas_real.h"
#include "cblas_real.h"
//#include "curand.h"
#include <vector>

//数据是否存储于CUDA设备
typedef enum
{
    CUDA_CPU = 0,
    CUDA_GPU,
} CudaType;

//该类包含一些cuda的基本参数，例如cublas和cudnn的handle
//此类型不能被随机创建，而仅能从已知的对象中选择一个
class CudaToolkit
{
public:
    friend class Matrix;
    friend class MatrixExtend;

public:
    CudaToolkit();      //请勿直接使用
    ~CudaToolkit();     //请勿直接使用

public:
    static CudaToolkit* select(int dev_id);
    static int checkDevices();

public:
    bool inited_ = false;
    //cublasHandle_t cublas_handle_ = nullptr;
    Cublas* cublas_ = nullptr;
    //Cblas* cblas_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    //curandGenerator_t curand_generator_ = nullptr;

    //这些是公用的，matrix类自带的是私用的
    cudnnTensorDescriptor_t tensor_desc_ = nullptr;
    cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
    cudnnConvolutionDescriptor_t convolution_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;

    cudnnActivationDescriptor_t activation_desc_ = nullptr;
    cudnnOpTensorDescriptor_t op_tensor_desc_ = nullptr;

    cudnnRNNDescriptor_t rnn_desc_ = nullptr;
    cudnnDropoutDescriptor_t dropout_desc_ = nullptr;
    cudnnSpatialTransformerDescriptor_t spatial_transformer_desc_ = nullptr;
    cudnnLRNDescriptor_t lrn_desc_ = nullptr;

    int nvml_id_;

private:
    static int device_count_;
    static std::vector<int> best_device_;
    static std::vector<int> nvml_cuda_id_;
    static std::vector<int> cuda_nvml_id_;
    static std::vector<CudaToolkit> cuda_toolkit_vector_;
    static CudaType global_cuda_type_;

public:
    int init(int use_cuda, int dev_id = -1);
    void destroy();
    static void destroyAll();
    static CudaType getCudaState() { return global_cuda_type_; }
    static void setCudaState(CudaType ct) { global_cuda_type_ = ct; }
    static void setDevice(int dev_id);  //仅为快速切换，无错误检查！
    void setDevice();
    static int getCurrentDevice();
    static int getBestDevice() { if (best_device_.size() <= 0) { findBestDevice(); } return best_device_[0]; }
    static int getBestDevices(int i = 0) { return best_device_.size() > i ? best_device_[i] : 0; }

    static int getCudaDeviceFromNvml(int nvml_id) { return nvml_cuda_id_[nvml_id]; }

    //设置张量数据，用于简化代码
    static void setTensorDesc(cudnnTensorDescriptor_t tensor, int w, int h, int c, int n)
    { if (tensor) { cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, n, c, h, w); } }

    //设置激活函数，用于简化代码
    static void setActivationDesc(cudnnActivationDescriptor_t activation, cudnnActivationMode_t mode, real v)
    { if (activation) { cudnnSetActivationDescriptor(activation, mode, CUDNN_NOT_PROPAGATE_NAN, v); } }

private:
    //copy and modify from helper_cuda.h
    static void findBestDevice();

public:
    //重载cudnn中Descriptor名字相似的一大堆函数方便使用，避免问题不使用模板
#ifndef _NO_CUDA
#define CUDNN_CERATE_DESCRIPTOR(type) \
    inline static cudnnStatus_t cudnnCreateDescriptor(cudnn##type##Descriptor_t* t) \
    { return cudnnCreate##type##Descriptor(t); }
#define CUDNN_DESTROY_DESCRIPTOR(type) \
    inline static cudnnStatus_t cudnnDestroyDescriptor(cudnn##type##Descriptor_t t) \
    { auto r = CUDNN_STATUS_EXECUTION_FAILED; if (t) { r=cudnnDestroy##type##Descriptor(t); t=nullptr; } return r; }
#define CUDNN_DESCRIPTOR_PAIR(type) \
    CUDNN_CERATE_DESCRIPTOR(type) CUDNN_DESTROY_DESCRIPTOR(type)

    CUDNN_DESCRIPTOR_PAIR(Tensor)
    CUDNN_DESCRIPTOR_PAIR(Pooling)
    CUDNN_DESCRIPTOR_PAIR(Convolution)
    CUDNN_DESCRIPTOR_PAIR(Filter)

    CUDNN_DESCRIPTOR_PAIR(Activation)
    CUDNN_DESCRIPTOR_PAIR(OpTensor)

#if CUDNN_VERSION >= 4000
    CUDNN_DESCRIPTOR_PAIR(RNN)
    CUDNN_DESCRIPTOR_PAIR(Dropout)
    CUDNN_DESCRIPTOR_PAIR(SpatialTransformer)
    CUDNN_DESCRIPTOR_PAIR(LRN)
#endif

#undef CUDNN_CERATE_DESCRIPTOR
#undef CUDNN_DESTROY_DESCRIPTOR
#undef CUDNN_DESCRIPTOR_PAIR
#endif

    static int cudnnCreateDescriptor(int**) { return 0; }
    static int cudnnDestroyDescriptor(int*) { return 0; }

    //Blas* selectBlas(CudaType mc) { return mc == mc_NoCuda ? (Blas*)(cblas) : (Blas*)(cublas); }
};



