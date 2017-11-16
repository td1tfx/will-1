#include "DataPreparer.h"
#include <time.h>
#include "MNIST.h"
#include "File.h"
#include "DataPreparerCreator.h"

DataPreparer::DataPreparer()
{
    rand_.set_seed();
    rand_.set_parameter(0, 1);
    srand(time(NULL));
}

DataPreparer::~DataPreparer()
{
}

void DataPreparer::init()
{
    //如果需要实时生成训练集，则可能要重定
    trans_ = option_->getIntFromSection(section_, "trans", 0);
    fill_ = option_->getIntFromSection(section_, "fill", 0);
    aem_ = option_->getInt("aem", 0);

    init2();
}

//初始化训练集准备器
int DataPreparer::getTrainDataForceGroup()
{
    if (fill_)
    {
        return option_->getIntFromSection(section_, "force_group", -1);
    }
    return -1;
}

//变换一组数据，并将其放入另一组数据集
//fill_queue表示data1中的第i个将被填入data0中的fill_queue[i]的数据
void DataPreparer::transData(DataGroup& data0, DataGroup& data1, const std::vector<int>& fill_queue)
{
    rand_.set_seed();
    srand(time(NULL));
    #pragma omp parallel for
    for (int i = 0; i < fill_queue.size(); i++)
    {
        auto temp0 = new Matrix(w0_, h0_, c0_, 1, MATRIX_DATA_OUTSIDE, CUDA_CPU);
        auto temp1 = new Matrix(temp0, MATRIX_DATA_OUTSIDE, CUDA_CPU);

        temp0->shareData(data0.X(), 0, fill_queue[i]);
        temp1->shareData(data1.X(), 0, i);

        //先取得该图的label，在后面参考
        int label = 0;
        for (int l = 0; l < data1.Y()->getRow(); l++)
        {
            if (data0.Y()->getData(l, fill_queue[i]) == 1)
            {
                label = l;
                break;
            }
        }
        transOne(temp0, temp1, label);
        delete temp0;
        delete temp1;
        if (aem_ != 1)
        {
            Matrix::copyDataPointer(data0.Y(), data0.Y()->getDataPointer(0, fill_queue[i]), data1.Y(), data1.Y()->getDataPointer(0, i), data1.Y()->getRow());
        }
    }
    //如果aem为1，则是完全自编码
    //如果aem为其他非零值，则输入的变化并不会被复制到输出部分，可以认为是降噪或者还原自编码
    if (aem_ == 1) { Matrix::copyData(data1.X(), data1.Y()); }
}

void DataPreparer::transOne(Matrix* matrix0, Matrix* matrix1, int label /*= 0*/)
{
    Matrix::copyData(matrix0, matrix1);
}

void DataPreparer::shuffleQueue(std::vector<int>& train_queue)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(train_queue_origin_.begin(), train_queue_origin_.end(), g);
}

void DataPreparer::reload()
{
    //safe_delete(train_data_preparer_);
    //train_data_preparer_ = DataPreparer::createByOption(option_);
}

//数据准备器将origin中的数据打乱，进行变换之后上传至data
void DataPreparer::prepareData(int epoch, DataGroup& origin, DataGroup& data)
{
    int ep1 = epoch + 1;

    if (train_queue_origin_.size() != origin.getNumber())
    {
        train_queue_origin_.resize(origin.getNumber());
        for (int i = 0; i < train_queue_origin_.size(); i++) { train_queue_origin_[i] = i; }
    }

    std::vector<int> train_queue_cpu(data.getNumber());
    bool need_refill = false;
    //复制训练序列origin到cpu
    int i = 0;
    while (i < train_queue_cpu.size())
    {
        if (trained_in_origin_ >= train_queue_origin_.size())
        {
            trained_in_origin_ = 0;
        }
        //如果出现越界就说明需要处理了
        if (trained_in_origin_ == 0)
        {
            LOG("Shuffle train queue for epoch %d\n", ep1);
            shuffleQueue(train_queue_origin_);
            need_refill = true;
        }
        train_queue_cpu[i] = train_queue_origin_[trained_in_origin_];
        i++;
        trained_in_origin_++;
    }
    //重新采集train，其实严格说应该在循环内部采集，因为变换在后面
    if (need_refill && fill_)
    {
        LOG("Fill data for epoch %d\n", ep1);
        fillData(origin);
    }

    //可能包含对训练数据的变换
    if (&origin != &data)
    {
        if (trans_)
        {
            LOG("Transfer train data for epoch %d\n", ep1);
            transData(origin, data, train_queue_cpu);
        }
        else
        {
            //#pragma omp parallel for
            for (int i = 0; i < data.getNumber(); i++)
            {
                data.copyPartFrom(origin, train_queue_cpu[i], i, 1);
            }
        }
        LOG("Data prepared for epoch %d\n", ep1);
    }

    if (option_->getIntFromSection(section_, "save_checkpoint_data", 0))
    {
        auto fln = "check_point_" + std::to_string(epoch) + "_X.bin";
        auto fln1 = "check_point_" + std::to_string(epoch) + "_Y.bin";
        auto fln2 = "check_point_" + std::to_string(epoch) + "_d.txt";
        auto save_matrix = [](std::string out, Matrix * X)
        {
            FILE* pf = fopen(out.c_str(), "w+b");
            X->save(pf);
            fclose(pf);
        };
        save_matrix(fln, data.X());
        save_matrix(fln1, data.Y());
        FILE* pf = fopen(fln2.c_str(), "w+");
        fprintf(pf, "%d %d %d %d\n", data.X()->getWidth(), data.X()->getHeight(), data.X()->getChannel(), data.X()->getNumber());
        fclose(pf);
    }
}

//读取数据
void DataPreparer::read(DataGroup& train, DataGroup& test)
{

    if (option_->getInt("use_mnist", 0) == 0)
    {
        int data_in_txt = option_->getInt("data_in_txt", 0);
        if (data_in_txt)
        {
            readTxt(option_->getString("TrainDataFile").c_str(), train);
            readTxt(option_->getString("TestDataFile").c_str(), test);
        }
        else
        {
            readBin(option_->getString("TrainDataLabelFile"), option_->getString("TrainDataBinFile"), train);
            readBin(option_->getString("TestDataLabelFile"), option_->getString("TestDataBinFile"), test);
        }
        if (test.Y())
        {
            test.createA();
        }
    }
    else
    {
        //使用MNIST库，通常用来测试网络
        MNIST mnist;
        std::string mnist_path = option_->getString("mnist_path", "mnist");
        mnist.load(train, test, mnist_path);
    }

    //自编码器则将X复制到Y
    //去噪自编码等不属于严格自编码器，请自行处理相关数据
    if (aem_ == 1)
    {
        train.setY(train.X()->clone(MATRIX_DATA_INSIDE, CUDA_CPU));
        if (test.X())
        {
            test.setY(test.X()->clone(MATRIX_DATA_INSIDE, CUDA_CPU));
        }
    }
}

//从txt文件读取数据到DataGroup
//该函数只读取到CPU，如需读取至GPU请调用后再写一步
//这里的处理可能不是很好
void DataPreparer::readTxt(const std::string& filename, DataGroup& data)
{
    int count = 0;
    if (filename == "")
    {
        setNullData(data);
        return;
    }

    int mark = 3;
    //数据格式：前两个是输入变量数和输出变量数，之后依次是每组的输入和输出，是否有回车不重要
    std::string str = convert::readStringFromFile(filename);
    if (str == "") { return; }
    std::vector<real> v;
    int n = convert::findNumbers(str, &v);
    if (n <= 0) { return; }

    int x_size = int(v[0]);
    int y_size = int(v[1]);

    count = (n - mark) / (x_size + y_size);
    data.clear();
    data.setX(new Matrix(x_size, count, MATRIX_DATA_INSIDE, CUDA_CPU));
    data.setY(new Matrix(y_size, count, MATRIX_DATA_INSIDE, CUDA_CPU));

    //写法太难看了
    int k = mark, k1 = 0, k2 = 0;

    for (int i_data = 1; i_data <= count; i_data++)
    {
        for (int i = 1; i <= x_size; i++)
        {
            data.X()->getData(k1++) = v[k++];
        }
        for (int i = 1; i <= y_size; i++)
        {
            data.Y()->getData(k2++) = v[k++];
        }
    }
}

//从bin文件读取数据到DataGroup
//该函数只读取到CPU，如需读取至GPU请调用后再写一步
void DataPreparer::readBin(const std::string& file_txt, const std::string& file_bin, DataGroup& data)
{
    if (file_txt == "")
    {
        setNullData(data);
        return;
    }
    auto data_bin = File::readFile(file_bin.c_str());

    //二进制数据文件定义：宽，高，通道，图片数，数据
    int w = *(int*)(data_bin);
    int h = *(int*)(data_bin + 4);
    int c = *(int*)(data_bin + 8);
    int n = *(int*)(data_bin + 12);

    LOG("Read bin file, w = %d, h = %d, c = %d, n = %d\n", w, h, c, n);

    //*group = n;

    int64_t total_size = w * h * c * n;
    data.clear();

    data.setX(new Matrix(w, h, c, n, MATRIX_DATA_INSIDE, CUDA_CPU));

    for (int64_t i = 0; i < total_size; i++)
    {
        auto v = *(unsigned char*)(data_bin + 16 + i) / 255.0;
        data.X()->getData(i) = v;
    }

    //以下处理Y，如果是AEM的话，不需要处理
    if (aem_ == 0)
    {
        data.setY(new Matrix(w1_, h1_, c1_, n, MATRIX_DATA_INSIDE, CUDA_CPU));
        data.Y()->initData(0);
        FILE* fid_txt = fopen(file_txt.c_str(), "rt");

        char s[4096];
        int label;
        if (fid_txt)
        {
            for (int i = 0; i < n; i++)
            {
                //fgets(s, 4096, fid_txt);
                //auto strs = convert::splitString(s, " ");
                //if (strs.size() >= 2)
                //{
                //label = atoi(strs[1].c_str());
                //}
                label = 0;
                fscanf(fid_txt, "%s %d", s, &label);
                //label = s[0] - 0x30;
                if (data.Y()->getRow() == 1)
                {
                    data.Y()->getData(i) = label;
                }
                else
                {
                    //LOG("%d", label);
                    data.Y()->getData(label, i) = 1;
                }
            }
            fclose(fid_txt);
        }
    }
}

//若读取数据失败，依据网络的输入和输出创建一个，避免后续的错误，无太大的意义
void DataPreparer::setNullData(DataGroup& data)
{
    data.setX(new Matrix(w0_, h0_, c0_, n_, MATRIX_DATA_INSIDE, CUDA_CPU));
    data.setY(new Matrix(w1_, h1_, c1_, n_, MATRIX_DATA_INSIDE, CUDA_CPU));
}

void DataPreparer::resizeDataGroup(DataGroup& data)
{
    if (fill_)
    {
        int force_train_group = getTrainDataForceGroup();
        if (data.X() && force_train_group > 0)
        {
            data.resize(force_train_group);
        }
    }
}

void DataPreparer::setWHC(int w0, int h0, int c0, int w1, int h1, int c1)
{
    w0_ = w0;
    h0_ = h0;
    c0_ = c0;
    w1_ = w1;
    h1_ = h1;
    c1_ = c1;
}

std::string DataPreparer::getMessage(int i)
{
    if (i >= 0 && i < message_.size()) { return message_[i]; }
    return "Error.";
}

