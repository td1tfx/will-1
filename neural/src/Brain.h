#pragma once
#include <cstdio>
#include <vector>
#include <cstring>
#include <cmath>
#include "DataPreparerCreator.h"
#include <thread>
#include "DataGroup.h"
#include "Timer.h"
#include "Net.h"
#include <atomic>

//总调度
class Brain : public Neural
{
public:
    Brain();
    virtual ~Brain();

public:
    int brain_id_;

protected:
    int max_batch_ = 100000;  //一次能处理的数据量，与内存或显存大小相关
    int batch_;

    std::function<void(Brain*)> running_callback_ = nullptr;  //回调函数

    Option* option_ = nullptr;
    Timer timer_total_;
    int MP_count_ = 1;
    std::vector<Net*> nets_;
    WorkModeType work_mode_ = WORK_MODE_NORMAL;

public:
    void setOptions(const std::string& name_values) { option_->setOptions(option_->getDefautlSection(), name_values); }
    void setOptions(const std::string& section, const std::string& name_values) { option_->setOptions(section, name_values); }
    void setOptions(const std::string& section, std::vector<std::string> name_values) { option_->setOptions(section, name_values); }
    //载入ini文件，注意实际上可以载入多个
    void load(const std::string& ini_file) { option_->loadIniFile(ini_file); }
    void setCallback(std::function<void(Brain*)> f) { running_callback_ = f; }

    Option* getOption() { return option_; }
protected:
    //epoch和iter的统计
    int epoch_count_ = 0;
    int iter_count_ = 0;

public:
    int getIterCount() { return iter_count_; }
    void setIterCount(int ic) { iter_count_ = ic; }

public:
    //real scale_data_x_ = 1;   //x参与计算之前先乘以此数，太麻烦，不搞了
    real train_accuracy_ = 0; //训练集上准确率
    real test_accuracy_ = 0;  //测试集上准确率

public:
    //训练集
    //这里采取的是变换之后将数据传入显存的办法，每个epoch都需要有这一步
    //这里理论上会降低速度，如果在显存里保存两份数据交替使用可以增加效率
    //此处并未使用，给GPU一些休息的时间

    //原始数据组数为cpu的数据的整数倍，通常为1倍，准备器中train_queue的变换保证数据被选中的机会一致
    //如果每次原始数据都是即时生成，算法并无变化，但最好设置为等同于cpu数据组数，以使数据充分被利用

    //DataGroup train_data_;         //无后缀保存于GPU中，专用于计算 -- 移除改为局部变量
    DataGroup train_data_origin_;    //origin读取的结果，即原始数据，但是可能会被进行一些变换，打乱顺序增加干扰等
    DataGroup train_data_cpu_;       //mirror是图像变换和顺序调整后的结果，会直接上传至GPU
    DataGroup test_data_origin_;     //原始测试集
    DataGroup test_data_cpu_;        //经变换后的测试集

    DataPreparer* data_preparer_ = nullptr; // 准备器
    DataPreparer* getDataPreparer() { return data_preparer_; }
protected:
    void initNets();
    virtual void initDataPreparer();
    void initData();
public:
    void train(std::vector<Net*> nets, DataPreparer* data_preparer, int epoches, int test_epoch = 1);
    void train(Net* net, DataPreparer* data_preparer, int epoches, int test_epoch = 1) { train({ net }, data_preparer, epoches, test_epoch); }
private:
    //在主线程进行数据准备，副线程进行训练和测试
    struct TrainInfo
    {
        std::atomic<int> data_prepared;                  //0 未准备好，1 cpu准备完毕， 2 gpu准备完毕
        std::atomic<int> data_distributed;               //已经复制过的线程
        std::atomic<int> stop;                           //结束信息
        std::atomic<int> trained;                        //已经训练完成的网络个数
        std::atomic<int> parameters_collected;           //数据同步完毕
        void reset()
        {
            data_prepared = 0;
            data_distributed = 0;
            stop = 0;
            trained = 0;
            parameters_collected = 0;
        }
        TrainInfo() { reset(); }
        ~TrainInfo() {}
    };
    //注意，这里只能用宏，用函数写起来很麻烦
#define WAIT_UNTIL(condition) { while(!(condition)) { std::this_thread::sleep_for(std::chrono::nanoseconds(100)); } }
#define WAIT_UNTIL_OVERTIME(condition, overtime)\
    {\
        auto t0 = std::chrono::system_clock::now();\
        while(!(condition && std::chrono::system_clock::now() - t0 < std::chrono::nanoseconds(long long(overtime))))\
        { std::this_thread::sleep_for(std::chrono::nanoseconds(100)); }\
    }
    void trainOneNet(std::vector<Net*> nets, int net_id, TrainInfo* train_info, int epoch0, int epoches);
public:
    Net* getNet(int i = 0) { if (nets_.size() > i) { return nets_[i]; } return nullptr; }   //应注意当前的gpu

public:
    //以下构成一组调度范例
    int init(const std::string& ini_file = "");
    void run(int train_epoches = -1);
    void testOrigin(Net* net, int force_output = 0, int test_max = 0);
    void extraTest(Net* net, const std::string& filename, int force_output = 0, int test_max = 0);

public:
    void testData(real* x, int w0, int h0, int c0, int n, real* y, int w1, int h1, int c1);

public:
    void setLearnRateBase(real lrb) { for (auto& n : nets_) { n->setLearnRateBase(lrb); } }
};


