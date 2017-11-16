#include "CudaToolkit.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

int CudaToolkit::device_count_ = -1;    //-1表示没有初始化，该值正常的值应为非负

//以下的向量即使没有设备，也会被初始化为1个元素
std::vector<int> CudaToolkit::best_device_;
std::vector<int> CudaToolkit::nvml_cuda_id_;    //nvml和cudaSetDevice的取值不同，其中nvml的值应当更加合理，此处转换
std::vector<int> CudaToolkit::cuda_nvml_id_;    //获取当前设备用
std::vector<CudaToolkit> CudaToolkit::cuda_toolkit_vector_;
CudaType CudaToolkit::global_cuda_type_ = CUDA_CPU;

CudaToolkit::CudaToolkit()
{
}

CudaToolkit::~CudaToolkit()
{
    destroy();
}

CudaToolkit* CudaToolkit::select(int dev_id)
{
    if (global_cuda_type_ == CUDA_CPU)
    {
        if (device_count_ < 0)
        {
            cuda_toolkit_vector_.resize(1);
            device_count_ = 0;
        }
        return &cuda_toolkit_vector_[0];
    }
    else
    {
        //未初始化则尝试之
        if (device_count_ < 0)
        {
            checkDevices();
            if (device_count_ == 0) { return nullptr; }
        }

        if (dev_id < 0 || dev_id >= device_count_)
        {
            dev_id = getBestDevice();
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, nvml_cuda_id_[dev_id]);
            fprintf(stdout, "Auto choose the best device %d: \"%s\" with compute capability %d.%d\n",
                dev_id, device_prop.name, device_prop.major, device_prop.minor);
        }

        auto& cuda_tk = cuda_toolkit_vector_[dev_id];

        if (!cuda_tk.inited_)
        {
            cuda_tk.init(1, dev_id);
        }

        return &cuda_tk;
    }
}

//返回值为设备数
int CudaToolkit::checkDevices()
{
    device_count_ = 0;
    if (cudaGetDeviceCount(&device_count_) != cudaSuccess || device_count_ <= 0) { return 0; }

    nvml_cuda_id_.resize(device_count_);
    cuda_nvml_id_.resize(device_count_);
    cuda_toolkit_vector_.resize(device_count_);

    std::vector<int> cuda_pci(device_count_), nvml_pci(device_count_);
    //首先指定一个假的
    for (int i = 0; i < device_count_; i++)
    {
        nvml_cuda_id_[i] = i;
        cuda_pci[i] = i;
        nvml_pci[i] = i;
    }

#ifdef NVML_API_VERSION
    nvmlInit();
    for (int i = 0; i < device_count_; i++)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        cuda_pci[i] = device_prop.pciBusID;

        nvmlDevice_t nvml_device;
        nvmlPciInfo_st nvml_pciinfo;
        nvmlDeviceGetHandleByIndex(i, &nvml_device);
        nvmlDeviceGetPciInfo(nvml_device, &nvml_pciinfo);
        nvml_pci[i] = nvml_pciinfo.bus;
    }
#endif
    //fprintf(stdout, "nvml(cuda) ");
    for (int nvml_i = 0; nvml_i < device_count_; nvml_i++)
    {
        for (int cuda_i = 0; cuda_i < device_count_; cuda_i++)
        {
            if (cuda_pci[cuda_i] == nvml_pci[nvml_i])
            {
                nvml_cuda_id_[nvml_i] = cuda_i;
                cuda_nvml_id_[cuda_i] = nvml_i;
            }
        }
        //fprintf(stdout, "%d(%d) ", nvml_i, nvml_cuda_id_[nvml_i]);
    }
    //fprintf(stdout, "\n");

    findBestDevice();
    return device_count_;
}

//返回0正常，其他情况都有问题
int CudaToolkit::init(int use_cuda, int dev_id /*= -1*/)
{
    //cblas_ = new Cblas();

    nvml_id_ = dev_id;
    cudnnCreateDescriptor(&tensor_desc_);
    cudnnCreateDescriptor(&activation_desc_);
    cudnnCreateDescriptor(&op_tensor_desc_);
    cudnnCreateDescriptor(&pooling_desc_);
    cudnnCreateDescriptor(&convolution_desc_);
    cudnnCreateDescriptor(&filter_desc_);
    cudnnCreateDescriptor(&rnn_desc_);
    cudnnCreateDescriptor(&dropout_desc_);
    cudnnCreateDescriptor(&spatial_transformer_desc_);
    cudnnCreateDescriptor(&lrn_desc_);

    cudaSetDevice(nvml_cuda_id_[dev_id]);
    cublas_ = new Cublas();
    if (cublas_->init() != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization error on device %d!\n", dev_id);
        return 1;
    }
    if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUDNN initialization error on device %d!\n", dev_id);
        return 2;
    }
    //if (curandCreateGenerator(&toolkit_.curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
    //{
    //    fprintf(stderr, "CURAND initialization error!\n");
    //    return 3;
    //}
    //else
    //{
    //    curandSetPseudoRandomGeneratorSeed(toolkit_.curand_generator_, 1234ULL);
    //}
    fprintf(stdout, "CUDA initialization on device %d succeed\n", nvml_id_);
    inited_ = true;
    global_cuda_type_ = CUDA_GPU;

    return 0;
}

void CudaToolkit::destroy()
{
#ifndef _NO_CUDA
    cudnnDestroyDescriptor(tensor_desc_);
    cudnnDestroyDescriptor(activation_desc_);
    cudnnDestroyDescriptor(op_tensor_desc_);
    cudnnDestroyDescriptor(pooling_desc_);
    cudnnDestroyDescriptor(convolution_desc_);
    cudnnDestroyDescriptor(filter_desc_);
    cudnnDestroyDescriptor(rnn_desc_);
    cudnnDestroyDescriptor(dropout_desc_);
    cudnnDestroyDescriptor(spatial_transformer_desc_);
    cudnnDestroyDescriptor(lrn_desc_);
#endif
    if (cublas_) { cublas_->destroy(); }
    if (cudnn_handle_) { cudnnDestroy(cudnn_handle_); }
    //if (cblas_) { delete cblas_; }
    global_cuda_type_ = CUDA_CPU;
}

void CudaToolkit::destroyAll()
{
    best_device_.clear();
    nvml_cuda_id_.clear();
    cuda_nvml_id_.clear();
    cuda_toolkit_vector_.clear();
#ifdef NVML_API_VERSION
    nvmlShutdown();
#endif
}

void CudaToolkit::setDevice(int dev_id)
{
    if (nvml_cuda_id_.size() > dev_id) { cudaSetDevice(nvml_cuda_id_[dev_id]); }
}

void CudaToolkit::setDevice()
{
    setDevice(nvml_id_);
}

int CudaToolkit::getCurrentDevice()
{
    int device;
    cudaGetDevice(&device);
    return cuda_nvml_id_[device];
}

// This function returns the best GPU (with maximum GFLOPS)
void CudaToolkit::findBestDevice()
{
    auto getSPcores = [](cudaDeviceProp & devive_prop) -> int
    {
        int cores = 0;
        int mp = devive_prop.multiProcessorCount;
        switch (devive_prop.major)
        {
        case 2: // Fermi
            if (devive_prop.minor == 1) { cores = mp * 48; }
            else { cores = mp * 32; }
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devive_prop.minor == 1) { cores = mp * 128; }
            else if (devive_prop.minor == 0) { cores = mp * 64; }
            else { fprintf(stderr, "Unknown device type\n"); }
            break;
        default:
            fprintf(stderr, "Unknown device type\n");
            break;
        }
        return cores;
    };

    if (device_count_ <= 0)
    {
        best_device_.resize(1);
        best_device_[0] = 0;
        return;
    }

    best_device_.resize(device_count_);
    int i = 0;
    cudaDeviceProp device_prop;

    double best_state = -1e8;
    std::vector<double> state(device_count_);
    std::vector<int> pci(device_count_);
    int best_i = 0;

    for (int i = 0; i < device_count_; i++)
    {
        state[i] = 0;
        pci[i] = i;
        best_device_[i] = i;
        cudaGetDeviceProperties(&device_prop, nvml_cuda_id_[i]);

        if (device_prop.computeMode != cudaComputeModeProhibited)
        {
            //通常情况系数是2，800M系列是1，但是不管了
            double flops = 2.0e3 * device_prop.clockRate * getSPcores(device_prop);
#ifdef NVML_API_VERSION
            nvmlDevice_t nvml_device;
            nvmlDeviceGetHandleByIndex(i, &nvml_device);
            nvmlMemory_t nvml_memory;
            nvmlDeviceGetMemoryInfo(nvml_device, &nvml_memory);
            nvmlPciInfo_st nvml_pciinfo;
            nvmlDeviceGetPciInfo(nvml_device, &nvml_pciinfo);
            pci[i] = nvml_pciinfo.bus;
            unsigned int temperature;
            nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temperature);
            fprintf(stdout, "Device %d(%d): %s, %7.2f GFLOPS, free memory %7.2f MB (%g%%), temperature %u C\n",
                i, nvml_cuda_id_[i], device_prop.name, flops / 1e9, nvml_memory.free / 1048576.0, 100.0 * nvml_memory.free / nvml_memory.total, temperature);
            state[i] = flops / 1e12 + nvml_memory.free / 1e9 - temperature / 5.0;
#else
            fprintf(stdout, "Device %d: %s, %7.2f GFLOPS, total memory %g MB\n",
                nvml_cuda_id_[i], device_prop.name, flops / 1e9, device_prop.totalGlobalMem / 1e6);
            state[i] += flops / 1e12 + device_prop.totalGlobalMem / 1e9;
#endif
            if (state[i] > best_state)
            {
                best_state = state[i];
                best_i = i;
            }
        }
    }

    //best_device_[0] = best_i;

    //这里需要重新计算，将所有设备按照当前状态排序，其中包括与最好设备的距离
    for (int i = 0; i < device_count_; i++)
    {
        state[i] -= std::abs((pci[i] - pci[best_i])) / 50.0;
    }

    for (int i = 0; i < device_count_ - 1; i++)
    {
        for (int j = 0; j < device_count_ - 1 - i; j++)
        {
            if (state[j] < state[j + 1])
            {
                std::swap(state[j], state[j + 1]);
                std::swap(best_device_[j], best_device_[j + 1]);
            }
        }
    }
}
