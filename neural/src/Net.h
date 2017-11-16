#pragma once
#include "Neural.h"
#include "Layer.h"
#include "DataGroup.h"

class Net : public Neural
{
public:
    Net();
    virtual ~Net();

private:
    Option* option_;

    //网络结构
    std::vector<Layer*> layer_vector_;      //此向量的顺序即计算顺序
    std::vector<Layer*> all_layer_vector_;  //全部查找到的层，包含一些多余的，析构时清理
    std::map<std::string, Layer*> all_layer_map_;  //按名字保存的层

    int device_id_ = 0;

    int batch_ = -1;
    int aem_ = 0;    //是否是自编码模式

    Matrix* parameters_ = nullptr;
    Matrix* workspace_ = nullptr;

    bool trained_ = false;    //该式标记网络里面的值是否是经过训练的，如果载入网络也会改动这个值

public:
    void setDevice(int dev) { CudaToolkit::setDevice(dev); device_id_ = dev; }
    int getDevice() { return device_id_; }
    void setDeviceSelf() { CudaToolkit::setDevice(device_id_); }
    Layer* getLayer(int number);
    Layer* getLayerByName(const std::string& name);
    Layer* getFirstLayer();
    Layer* getLastLayer();
    int getLayersCount() { return layer_vector_.size(); }
    std::vector<Layer*>& getLayerVector() { return layer_vector_; }
    Matrix*& getParameters() { return parameters_; }
    Matrix*& getWorkspace() { return workspace_; }
    void setOption(Option* op) { option_ = op; }
    bool hasTrained() { return trained_; }
    void setBatch(int batch) { batch_ = batch; }
    void init(const std::string& in_name = "layer_in", const std::string& out_name = "layer_out");

private:
    std::string layer_in_name_;
    std::string layer_out_name_;
    int createAndConnectLayers();

public:
    void active(Matrix* X, Matrix* Y, Matrix* A, bool learn, bool generate, real* error);
    void active(DataGroup& data, bool learn, bool generate, real* error) { active(data.X(), data.Y(), data.A(), learn, generate, error); }
    void moveDataPointer(int p);

    void save(const std::string& filename);
    void load(const std::string& filename);

    void calNorm(real& l1, real& l2);

private:
    void malloc(int batch_size);   //初始化网络的数据
    int resetBatchSize(int n);

public:
    real adjustLearnRate(int ec);
    void setActivePhase(ActivePhaseType ap) { for (auto& l : layer_vector_) { l->getActiver()->setActivePhase(ap); } }
    real test(const std::string& info, DataGroup& data, int force_output, int test_max);

public:
    void setLearnRateBase(real lrb);

};

