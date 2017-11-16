#include "Brain.h"
#include "Random.h"
#include "others/libconvert.h"
#include <algorithm>
#include "File.h"
#include "Color.h"

Brain::Brain()
{
    option_ = new Option();
}

Brain::~Brain()
{
    safe_delete(option_);
    safe_delete(nets_);
    safe_delete(data_preparer_);
}

//返回为0是正确创建
int Brain::init(const std::string& ini_file)
{
    LOG("%s\n", Timer::getNowAsString().c_str());

    //LOG("Size of real is %lu bytes\n", sizeof(real));

    //初始化选项
    if (ini_file != "")
    {
        option_->loadIniFile(ini_file);
    }
    //option_->print();

    //gpu测试
    int device_count = 0;
    if (option_->getInt("use_cuda", 1))
    {
        device_count = CudaToolkit::checkDevices();
        if (device_count > 0)
        {
            LOG("Found %d CUDA device(s)\n", device_count);
            CudaToolkit::setCudaState(CUDA_GPU);
        }
        else
        {
            LOG("No CUDA devices!!\n");
            CudaToolkit::setCudaState(CUDA_CPU);
            safe_delete(option_);
            return 1;
        }
    }

    if (option_->getInt("use_cuda") != 0 && CudaToolkit::getCudaState() != CUDA_GPU)
    {
        LOG("CUDA state is not right, refuse to run!\n");
        LOG("Re-init the net again, or consider CPU mode (slow).\n");
        return 1;
    }


    MP_count_ = std::min(device_count, option_->getInt("mp", 1));
    if (MP_count_ <= 0) { MP_count_ = 1; }
    batch_ = std::max(1, option_->getInt("batch", 100));

    work_mode_ = option_->getWorkModeType("work_mode", "normal");

    initNets();
    initDataPreparer();
    initData();

    return 0;
}

void Brain::initNets()
{
    nets_.resize(MP_count_);
    //这里读取ini中指定的device顺序，其中第一个设备为主网络，该值一般来说应由用户指定
    std::vector<int> mp_device(MP_count_);
    convert::findNumbers(option_->getString("mp_device"), &mp_device);
    //如果用户指定的不正确，则以best_device的顺序决定
    auto check_repeat = [](std::vector<int> v)
    {
        for (auto& a : v)
        {
            for (auto& b : v)
            {
                if (a == b) { return true; }
            }
        }
        return false;
    };
    if (mp_device.size() < MP_count_ || check_repeat(mp_device))
    {
        for (int i = 0; i < mp_device.size(); i++)
        {
            mp_device[i] = CudaToolkit::getBestDevices(i);
        }
    }

    for (int i = 0; i < MP_count_; i++)
    {
        auto net = new Net();
        net->setDevice(mp_device[i]);
        LOG("Net %d will be created on device %d\n", i, net->getDevice());
        CudaToolkit::select(mp_device[i]);
        net->setOption(option_);
        net->setBatch(batch_ / MP_count_);
        net->init();
        nets_[i] = net;
    }
    //只使用0号网络的权值
    for (int i = 1; i < nets_.size(); i++)
    {
        auto& net = nets_[i];
        if (!net->hasTrained())
        {
            Matrix::copyDataAcrossDevice(nets_[0]->getParameters(), net->getParameters());
        }
    }
    //主线程使用0号网络
    nets_[0]->setDeviceSelf();
}

void Brain::initDataPreparer()
{
    //数据准备器
    //这里使用公共数据准备器，实际上完全可以创建私有的准备器
    int w0, h0, c0, w1, h1, c1;
    nets_[0]->getFirstLayer()->getOutputSize(w0, h0, c0);
    nets_[0]->getLastLayer()->getOutputSize(w1, h1, c1);
    data_preparer_ = DataPreparerCreator::create(option_, "data_preparer", w0, h0, c0, w1, h1, c1);
    //aem_ = option_->getInt("aem", 0);
    data_preparer_->read(train_data_origin_, test_data_origin_);
    data_preparer_->resizeDataGroup(train_data_origin_);
}

//初始化训练集，必须在Prepare之后
void Brain::initData()
{
    //训练数据使用的显存量，不能写得太小
    double size_gpu = option_->getReal("cuda_max_train_space", 1e9);
    //计算显存可以放多少组数据，为了方便计算，要求是minibatch的整数倍，且可整除原始数据组数

    if (train_data_origin_.X())
    {
        //计算最大可以放多少组
        max_batch_ = floor(size_gpu / (train_data_origin_.X()->getRow() * sizeof(real) * batch_)) * batch_;
        LOG("max batch in video memory is %d\n", max_batch_);
        max_batch_ = std::min(train_data_origin_.X()->getNumber(), max_batch_);
        train_data_cpu_.initWithReference(train_data_origin_, max_batch_, MATRIX_DATA_INSIDE, CUDA_CPU);
        //data_preparer_->prepareData(0, train_data_origin_, train_data_cpu_);
    }

    //生成测试集，注意这里测试准备器所占用的内存会被释放，测试集通常只生成一次即可
    std::string test_section = "data_preparer_test";
    if (option_->getInt("test_test"))
    {
        /*if (!option_->hasSection(test_section))
        {
            option_->setOption(test_section, "test", "1");
        }*/
        auto data_preparer_test = DataPreparerCreator::createByReference(option_, test_section, data_preparer_);
        data_preparer_test->resizeDataGroup(test_data_origin_);
        test_data_cpu_.initWithReference(test_data_origin_, test_data_origin_.getNumber(), MATRIX_DATA_INSIDE, CUDA_CPU);
        data_preparer_test->prepareData(0, test_data_origin_, test_data_cpu_);
        safe_delete(data_preparer_test);
    }

    if (train_data_origin_.exist())
    {
        LOG("%d groups of train data\n", train_data_origin_.getNumber());
    }
    if (test_data_origin_.exist())
    {
        LOG("%d groups of test data\n", test_data_origin_.getNumber());
    }
}

//运行，注意容错保护较弱
//注意通常情况下是使用第一个网络测试数据
void Brain::run(int train_epoches /*= -1*/)
{
    auto net = nets_[0];
    //初测
    testOrigin(net, option_->getInt("force_output"), option_->getInt("test_max"));

    if (train_epoches < 0)
    {
        train_epoches = option_->getInt("train_epoches", 20);
    }
    LOG("Running for %d epoches...\n", train_epoches);

    train(nets_, data_preparer_, train_epoches, option_->getInt("test_epoch", 1));

    std::string save_filename = option_->getString("SaveFile");
    if (save_filename != "")
    {
        net->save(save_filename);
    }

    //终测
    testOrigin(net, option_->getInt("force_output"), option_->getInt("test_max"));
    //附加测试，有多少个都能用
    extraTest(net, option_->getString("extra_test_data_file"), option_->getInt("force_output"), option_->getInt("test_max"));

    LOG("Run neural net end. Elapsed time is %g s.\n", timer_total_.getElapsedTime());
    LOG("%s\n", Timer::getNowAsString().c_str());
}

//训练一批数据，输出步数和误差，若训练次数为0可以理解为纯测试模式
//首个参数为指定几个结构完全相同的网络并行训练
void Brain::train(std::vector<Net*> nets, DataPreparer* data_preparer, int epoches, int test_epoch /*= 1*/)
{
    if (epoches <= 0) { return; }

    int iter_per_epoch = train_data_origin_.X()->getNumber() / batch_;     //如果不能整除，则估计会不准确，但是关系不大
    if (iter_per_epoch <= 0) { iter_per_epoch = 1; }
    epoch_count_ = iter_count_ / iter_per_epoch;

    real e = 0, e0 = 0;

    Timer timer;
    //prepareData();
    TrainInfo train_info;
    train_info.data_prepared = 0;

    //创建训练进程
    std::vector<std::thread*> net_threads(nets.size());
    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i] = new std::thread{ &Brain::trainOneNet, this, nets, i, &train_info, epoch_count_, epoches };
    }

    train_info.stop = 0;
    int epoch0 = epoch_count_;
    for (int epoch_count = epoch0; epoch_count < epoch0 + epoches; epoch_count++)
    {
        data_preparer->prepareData(epoch_count, train_data_origin_, train_data_cpu_);
        train_info.data_prepared = 1;
        WAIT_UNTIL(train_info.data_distributed == MP_count_ || train_info.stop == 1);
        train_info.data_prepared = 0;
        train_info.data_distributed = 0;
        //回调
        if (running_callback_) { running_callback_(this); }
        iter_count_ += iter_per_epoch;
        epoch_count_++;
        LOG("%g s elapsed from previous check point, %g s elapsed totally\n", timer.getElapsedTime(), timer_total_.getElapsedTime());
        timer.start();
        if (train_info.stop == 1) { break; }
    }
    train_info.stop = 1;

    for (int i = 0; i < net_threads.size(); i++)
    {
        net_threads[i]->join();
    }
    safe_delete(net_threads);

    LOG("%g s elapsed from previous check point, %g s elapsed totally\n", timer.getElapsedTime(), timer_total_.getElapsedTime());
    LOG("\n");
}

//训练网络数组nets中的一个
void Brain::trainOneNet(std::vector<Net*> nets, int net_id, TrainInfo* train_info, int epoch0, int epoches)
{
    auto net = nets[net_id];
    net->setDeviceSelf();
    net->setActivePhase(ACTIVE_PHASE_TRAIN);

    DataGroup train_data_gpu, train_data_sub;
    train_data_gpu.initWithReference(train_data_cpu_, max_batch_ / MP_count_, MATRIX_DATA_INSIDE, CUDA_GPU);
    train_data_gpu.createA();
    train_data_sub.initWithReference(train_data_cpu_, batch_ / MP_count_, MATRIX_DATA_OUTSIDE, CUDA_GPU);
    //LOG("%g, %g\n", train_data_origin_.X()->dotSelf(), train_data_cpu_.Y()->dotSelf());

    DataGroup test_data_gpu;
    if (net_id == 0) { test_data_gpu.cloneFrom(test_data_cpu_); }

    int test_train = option_->getInt("test_train", 0);
    int test_train_origin = option_->getInt("test_train_origin", 0);
    int pre_test_train = option_->getInt("pre_test_train", 0);
    int test_test = option_->getInt("test_test", 0);
    int test_epoch = option_->getInt("test_epoch", 1);
    int save_epoch = option_->getInt("save_epoch", 10);
    int out_iter = option_->getInt("out_iter", 100);
    int test_max = option_->getInt("test_max", 0);
    std::string save_format = option_->getString("save_format", "save/save%d.txt");

    real max_test_accuracy = 0;
    int max_test_accuracy_epoch = 0;

    int iter_count = 0;
    int epoch_count = epoch0;
    while (epoch_count < epoch0 + epoches)
    {
        epoch_count++;
        //等待数据准备完成
        WAIT_UNTIL(train_info->data_prepared == 1 || train_info->stop == 1);
        train_data_gpu.copyPartFrom(train_data_cpu_, net_id * max_batch_ / MP_count_, 0, max_batch_ / MP_count_);
        //发出拷贝数据结束信号
        train_info->data_distributed++;
        //调整学习率
        real lr = net->adjustLearnRate(epoch_count);
        if (net_id == 0) { LOG("Learn rate of the last layer is %g\n", lr); }
        //训练前在训练集上的测试，若训练集实时生成可以使用
        if (net_id == 0 && epoch_count % test_epoch == 0 && pre_test_train) { train_accuracy_ = net->test("Pre-test on train set", train_data_gpu, 0, 1); }
        real e = 0;
        for (int iter = 0; iter < train_data_gpu.getNumber() / train_data_sub.getNumber(); iter++)
        {
            iter_count++;
            bool output = (iter + 1) % out_iter == 0;
            train_data_sub.shareData(train_data_gpu, iter * train_data_sub.getNumber());
            //LOG("%d, %g, %g\n", i, train_data_sub.X()->dotSelf(), train_data_sub.Y()->dotSelf());

            //同步未完成
            WAIT_UNTIL(train_info->parameters_collected == 0 || train_info->stop == 1);

            net->active(train_data_sub.X(), train_data_sub.Y(), nullptr, true, false, output ? &e : nullptr);
            //发出网络训练结束信号
            train_info->trained++;

            if (net_id == 0)
            {
                //主网络，完成信息输出，参数的收集和重新分发
                if (output)
                {
                    LOG("epoch = %d, iter = %d, error = %e\n", epoch_count, iter_count, e);
                }

                //主网络等待所有网络训练完成
                WAIT_UNTIL(train_info->trained == MP_count_ || train_info->stop == 1);
                train_info->trained = 0;
                //同步
                if (MP_count_ > 1)
                {
                    for (int i = 1; i < nets.size(); i++)
                    {
                        Matrix::copyDataAcrossDevice(nets[i]->getParameters(), net->getWorkspace());
                        Matrix::add(net->getParameters(), net->getWorkspace(), net->getParameters());
                    }
                    net->getParameters()->scale(1.0 / MP_count_);
                }
                //发布同步完成信号
                train_info->parameters_collected = MP_count_ - 1;
            }
            else
            {
                //非主网络等待同步结束
                WAIT_UNTIL(train_info->parameters_collected > 0 || train_info->stop == 1);
                train_info->parameters_collected--;
                //分发到各个网络
                Matrix::copyDataAcrossDevice(nets[0]->getParameters(), net->getParameters());
            }
        }
        if (net_id == 0) { LOG("Epoch %d finished.\n", epoch_count); }
        //主网络负责测试
        if (net_id == 0 && epoch_count % test_epoch == 0)
        {
            if (test_train != 0)
            {
                train_accuracy_ = net->test("Test on transformed train set", train_data_gpu, 0, test_max);
            }
            if (test_train_origin != 0)
            {
                train_accuracy_ = net->test("Test on original train set", train_data_origin_, 0, test_max);
            }
            if (test_test != 0)
            {
                test_accuracy_ = net->test("Test on transformed test set", test_data_gpu, 0, test_max);
                if (test_accuracy_ >= max_test_accuracy)
                {
                    max_test_accuracy = test_accuracy_;
                    max_test_accuracy_epoch = epoch_count;
                }
            }
            real l1, l2;
            net->calNorm(l1, l2);
            LOG("L1 = %g, L2 = %g\n", l1, l2);
            if (epoch_count % save_epoch == 0 && !save_format.empty())
            {
                //std::string name = convert::formatString("save%02d_%s.txt", epoch_count, Timer::getNowAsString().c_str());
                std::string save_name = convert::formatString(save_format.c_str(), epoch_count, test_accuracy_);
                //convert::replaceAllString(name, "\n", "");
                //convert::replaceAllString(name, " ", "_");
                //convert::replaceAllString(name, ":", "_");
                net->save(save_name);
            }
        }

        if (train_info->stop == 1)
        {
            break;
        }
    }

    if (net_id == 0 && test_test != 0)
    {
        Color::set(CONSOLE_COLOR_LIGHT_RED);
        LOG("Maximum accuracy on test set is %5.2f%% at epoch %d\n", max_test_accuracy * 100, max_test_accuracy_epoch);
        Color::set(CONSOLE_COLOR_NONE);
    }
}

//输出训练集和测试集的测试结果，注意这个测试仅在原始数据集上运行
void Brain::testOrigin(Net* net, int force_output /*= 0*/, int test_max /*= 0*/)
{
    if (net == nullptr) { net = nets_[0]; }
    if (option_->getInt("test_train") != 0)
    {
        train_accuracy_ = net->test("Test on original train set", train_data_origin_, force_output, test_max);
    }
    if (option_->getInt("test_test") != 0 && test_data_cpu_.X())
    {
        test_accuracy_ = net->test("Test on original test set", test_data_origin_, force_output, test_max);
    }
}

//附加测试集，一般无用
void Brain::extraTest(Net* net, const std::string& filename, int force_output /*= 0*/, int test_max /*= 0*/)
{
    if (filename == "") { return; }
    DataGroup extra_test_data;
    data_preparer_->readTxt(filename, extra_test_data);
    if (extra_test_data.exist())
    {
        net->test("Extra test", extra_test_data, force_output, test_max);
    }
}

void Brain::testData(real* x, int w0, int h0, int c0, int n, real* y, int w1, int h1, int c1)
{
    DataGroup data;
    data.setX(new Matrix(w0, h0, c0, n, x, CUDA_CPU));
    data.setY(new Matrix(w1, h1, c1, n, MATRIX_DATA_INSIDE, CUDA_CPU));
    data.createA();
    nets_[0]->test("Test data", data, 0, 0);
    auto temp_matrix = new Matrix(data.A(), MATRIX_DATA_OUTSIDE, CUDA_CPU);
    temp_matrix->shareData(y);
    Matrix::copyData(data.A(), temp_matrix);
    delete temp_matrix;
}





