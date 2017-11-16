#pragma once
#include <vector>
#include <functional>
#include <string>
#include "Option.h"
#include "Neural.h"
#include "MatrixExtend.h"
#include <thread>
#include "Modifier.h"
#include "Solver.h"
#include "Activer.h"


//神经层
//下面凡是有两个函数的，在无后缀函数中有公共部分，在带后缀函数中是各自子类的功能
class Layer : public Neural
{
public:
    Layer();
    virtual ~Layer();

protected:
    int id_;
    std::string layer_name_;  //本层的名字

    Neural* net_;

    Layer* prev_layer_ = nullptr;      //仅合并数据层可以有多个前层，其余层若多于一个会报警
    std::vector<Layer*> prev_layers_;  //除了合并数据层外，该向量仅用于生成网络结构，计算时无用
    std::vector<Layer*> next_layers_;  //后层可以有多个，它们获取到的是同样的数据

    Option* option_ = nullptr;

    int out_total_ = 1;  //对于全连接层，输出数等于节点数，对于其他形式定义不同。该值必须设定！
    int out_height_ = 1, out_width_ = 1, out_channel_ = 1; //图片模式时使用，对于其他模式，长和宽设为1

    //static int total_size_;    //总数据量
    int batch_ = 1;    //每批数据量

    int epoch_count_ = 0;
    int iter_count_ = 0;

    LayerVisibleType visible_type_ = LAYER_VISIBLE_HIDDEN;
    LayerConnectionType connetion_type_ = LAYER_CONNECTION_NONE;  //这个字段仅为标记，实际连接方式是虚继承

    bool need_train_ = true;   //是否需要训练，会影响前面的层

public:
    //前面一些字段的设置
    void setID(int id) { id_ = id; }
    void setNet(Neural* net) { net_ = net; }
    int getBatchSize() { return batch_; }
    void setBatchSize(int bs) { batch_ = bs; }
    bool getNeedTrain() { return need_train_; }
    void setNeedTrain(bool nt) { need_train_ = nt; }
    void setVisible(LayerVisibleType vt) { visible_type_ = vt; }
    LayerConnectionType getConnection() { return connetion_type_; }
    LayerConnectionType getConnection2();
    void setConnection(LayerConnectionType ct) { connetion_type_ = ct; }
    void getOutputSize(int& width, int& height, int& channel) { width = out_width_; height = out_height_; channel = out_channel_; }
    void setOption(Option* op) { option_ = op; }
    const std::string& getName() { return layer_name_; }
    void setName(const std::string& name);
    void setEpochCount(int ec) { epoch_count_ = ec; }

    int getOutWidth() { return out_width_; }
    int getOutHeight() { return out_height_; }
    int getOutChannel() { return out_channel_; }
    int getOutTotal() { return out_total_; }

public:
    //将连接作为列表返回，目的是适应数据转发层
    void addPrevLayers(Layer* layer) { prev_layer_ = layer; prev_layers_.push_back(layer); }
    void addNextLayers(Layer* layer) { next_layers_.push_back(layer); }

    Layer* getPrevLayer() { return prev_layer_;  }
    std::vector<Layer*> getPrevLayers() { return prev_layers_; }
    std::vector<Layer*> getNextLayers() { return next_layers_; }
    void clearConnect() { prev_layers_.clear(); next_layers_.clear(); prev_layer_ = nullptr; }
    int getNextLayersCount() { return next_layers_.size(); }
    Layer* getNextLayer(int i) { return next_layers_[i]; }

protected:
    //这几个矩阵形式相同，计算顺序： X, A, ..., dA, dX
    //当本层没有激活函数时，A与X指向同一对象，dA与dX指向同一对象
    Matrix* X_ = nullptr;    //X收集上一层的输出
    Matrix* dX_ = nullptr;
    Matrix* A_ = nullptr;    //激活函数作用之后就是本层输出A，输入层需要直接设置A
    Matrix* dA_ = nullptr;

    Matrix* Y_ = nullptr;    //Y相当于标准答案，仅输出层使用
    Matrix* dY_ = nullptr;   //dY的和除以组数就是误差
public:
    //以下函数仅建议使用在输入和输出层，隐藏层不建议使用！
    Matrix* getX() { return X_; }
    Matrix* getY() { return Y_; }
    Matrix* getA() { return A_; }
    Matrix* getDA() { return dA_; }
    real getAValue(int x, int y) { return A_->getData(x, y); }

protected:
    Matrix* W_ = nullptr;
    Matrix* dW_ = nullptr;

    bool need_bias_ = true;
    Matrix* b_ = nullptr;
    Matrix* db_ = nullptr;

    //Random<real> random_generator_;

public:
    Matrix*& getWeight() { return W_; }
    Matrix*& getBias() { return b_; }

    void init();

    //init2由子类处理，子类应计算本层输出的长宽通道和总数，以及初始化X和dX，如果有W和b也应同时初始化
    virtual void init2() {}
    //virtual void connect(std::vector<Layer*>& layers);
    void destroyData();
    void resetGroupCount();
    virtual void resetGroupCount2() {}

public:
    //基类的实现里只处理公共部分，不处理任何算法，即使算法有重复的部分仍然在子类处理！！

    //下面全部是计算用的函数，不是虚函数的尽量inline！！
    //以下Cost计算全部是输出层的！
    real calCostValue();
    void calCostDA();
    void calCostDX();

    //正向
    void activeForward();

    //反向
    void activeBackward();

    void updateABackward();

protected:
    //std::thread* thread_update_ = nullptr;
    //bool use_thread_ = false;
    virtual void updateX() {}
    void updateA();

    //用本层的dX更新上一层的dA
    //这里注意实际上后一层才能知道连接的情况，所以必须由后一层完成
    virtual void updatePrevLayerDA() {}

    void updateDA();
    void updateDX();

    //计算出参数的变化，一般是dW和dBias
    virtual void updateDParameters(real momentum) {}

private:
    Activer* activer_;
public:
    ActiveFunctionType getActiveFunction() { return activer_->getActiveFunction(); }
    Activer* getActiver() { return activer_; }

private:
    Solver* solver_;
public:
    real adjustLearnRate() { return solver_->adjustLearnRate(epoch_count_); }
    void updateParametersPre();   //在updateDParameters之前先更新一次，NAG使用
    void updateParameters();
    Solver* getSolver() { return solver_; }

private:
    Modifier* modifier_ = nullptr;

public:
    void save(FILE* fout);
    void load(FILE* fin);

public:
    void setPrune(int p) { prune_ = p; }
    std::vector<int> prune_record_;
protected:
    int prune_ = 0;
};
