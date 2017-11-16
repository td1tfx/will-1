#include "Net.h"
#include "LayerCreator.h"
#include <algorithm>
#include "Color.h"

Net::Net()
{
}


Net::~Net()
{
    safe_delete(all_layer_vector_);
}


Layer* Net::getLayer(int number)
{
    if (number >= 0 && number < layer_vector_.size())
    {
        return layer_vector_[number];
    }
    return nullptr;
}

Layer* Net::getLayerByName(const std::string& name)
{
    if (all_layer_map_.count(name) > 0)
    {
        return all_layer_map_[name];
    }
    return nullptr;
}


Layer* Net::getFirstLayer()
{
    if (!layer_vector_.empty())
    {
        return layer_vector_[0];
    }
    return nullptr;
}


Layer* Net::getLastLayer()
{
    if (!layer_vector_.empty())
    {
        return layer_vector_.back();
    }
    return nullptr;
}

//注意：输入和输出层的名字可以自定义，因此一个ini中可以写多个网络
void Net::init(const std::string& in_name, const std::string& out_name)
{
    layer_in_name_ = in_name;
    layer_out_name_ = out_name;
    //aem和batch在一开始会用来设置网络的初始大小
    aem_ = option_->getInt("aem", 0);
    setLog(option_->getInt("output_net", 1));
    createAndConnectLayers();
    setLog(1);
}

int Net::createAndConnectLayers()
{
    Layer* layer_in;
    Layer* layer_out;

    //lambda函数：计算上层和下层
    auto connect_layers = [&]()
    {
        for (auto& name_layer : all_layer_map_)
        {
            auto& l = name_layer.second;
            auto nexts = convert::splitString(option_->getStringFromSection(l->getName(), "next"), ",");
            for (auto& next : nexts)
            {
                convert::replaceAllString(next, " ", "");
                if (next != "" && all_layer_map_.count(next) > 0)
                {
                    l->addNextLayers(all_layer_map_[next]);
                    all_layer_map_[next]->addPrevLayers(l);
                }
            }
        }
    };
    //lambda函数：层是否已经在向量中
    auto contains = [&](std::vector<Layer*>& v, Layer * l) -> bool {return std::find(v.begin(), v.end(), l) != v.end(); };
    //lambda函数：递归将层压入向量
    //最后一个参数为假，仅计算是否存在连接，为真则是严格计算传导顺序
    std::function<void(Layer*, int, std::vector<Layer*>&, bool)> push_cal_stack = [&](Layer * layer, int direct, std::vector<Layer*>& stack, bool turn)
    {
        std::vector<Layer*> connect0, connect1;
        connect1 = layer->getNextLayers();
        connect0 = layer->getPrevLayers();

        if (direct < 0)
        {
            std::swap(connect0, connect1);
        }

        bool contain_all0 = true;
        for (auto& l : connect0)
        {
            if (!contains(stack, l))
            {
                contain_all0 = false;
                break;
            }
        }
        if (!turn || (!contains(stack, layer) && contain_all0))
        {
            stack.push_back(layer);
        }
        else
        {
            return;
        }
        for (auto& l : connect1)
        {
            push_cal_stack(l, direct, stack, turn);
        }
    };

    //查找所有存在定义的层
    auto sections = option_->getAllSections();

    //先把层都创建起来
    for (auto& section : sections)
    {
        if (section.find_first_of("layer") == 0)
        {
            LOG("Found layer %s\n", section.c_str());
            auto ct = option_->getLayerConnectionTypeFromSection(section, "type", "none");
            auto l = LayerCreator::createByConnectionType(ct);
            l->setOption(option_);
            l->setName(section);
            l->setNet(this);
            all_layer_vector_.push_back(l);
            if (l->getName() == layer_in_name_)
            {
                layer_in = l;
                l->setVisible(LAYER_VISIBLE_IN);
            }
            if (l->getName() == layer_out_name_)
            {
                layer_out = l;
                l->setVisible(LAYER_VISIBLE_OUT);
            }
            all_layer_map_[l->getName()] = l;
        }
    }

    int ret = 0;

    //连接，计算双向的连接，清除无用项，重新连接
    connect_layers();
    std::vector<Layer*> forward, backward;
    //层的计算顺序
    push_cal_stack(layer_in, 1, forward, false);
    push_cal_stack(layer_out, -1, backward, false);

    //不在双向中的都废弃
    //注意无效层的指针并没有清除，只占用很少的空间，在网络析构时处理
    std::vector<std::pair<std::string, Layer*>> temp;
    for (auto& name_layer : all_layer_map_)
    {
        if (!(contains(forward, name_layer.second) && contains(backward, name_layer.second)))
        {
            temp.push_back(name_layer);
        }
    }
    for (auto& name_layer : temp)
    {
        all_layer_map_.erase(name_layer.first);
        //safe_delete(name_layer.second);
        LOG("Remove bad layer %s\n", name_layer.first.c_str());
    }

    for (auto& l : all_layer_vector_)
    {
        l->clearConnect();
    }
    connect_layers();
    forward.clear();
    backward.clear();
    push_cal_stack(layer_in, 1, layer_vector_, true);
    for (int i = 0; i < layer_vector_.size(); i++)
    {
        layer_vector_[i]->setID(i);
    }
    //push_cal_stack(layer_out, -1, backward, true);

    //若不包含out项则有误
    if (contains(layer_vector_, layer_out))
    {
        //创建网络成功不表示没有问题，使用者应自己保证结构正确
        LOG("Net created\n");
    }
    else
    {
        ret = 1;
        LOG("Net create failed!\n");
    }

    malloc(batch_);

    return ret;
}

//learn为真时，会反向更新网络
//active只处理一个minibatch
//A是外部提供的矩阵，用于保存结果
void Net::active(Matrix* X, Matrix* Y, Matrix* A, bool learn /*= false*/, bool generate /*= false*/, real* error /*= nullptr*/)
{
    setDeviceSelf();

    getFirstLayer()->getA()->shareData(X);
    getLastLayer()->getY()->shareData(Y);

    //LOG("active %g, %g\n", getFirstLayer()->getA()->dotSelf(), getLastLayer()->getY()->dotSelf());

    for (int i_layer = 1; i_layer < getLayersCount(); i_layer++)
    {
        layer_vector_[i_layer]->activeForward();
    }
    if (learn)
    {
        for (int i_layer = getLayersCount() - 1; i_layer > 0; i_layer--)
        {
            if (layer_vector_[i_layer]->getNeedTrain())
            {
                layer_vector_[i_layer]->activeBackward();
            }
            else
            {
                //若某层不需要训练，前面的层也不需要训练，只需设置一个层即可
                break;
            }
        }
        for (int i_layer = getLayersCount() - 1; i_layer > 0; i_layer--)
        {
            if (layer_vector_[i_layer]->getNeedTrain())
            {
                layer_vector_[i_layer]->updateParameters();
            }
            else
            {
                //若某层不需要训练，前面的层也不需要训练，只需设置一个层即可
                break;
            }
        }
    }
    if (generate)
    {
        for (int i_layer = getLayersCount() - 1; i_layer >= 0; i_layer--)
        {
            layer_vector_[i_layer]->activeBackward();
        }
        getFirstLayer()->updateABackward();
    }
    if (A)
    {
        Matrix::copyDataPointer(getLastLayer()->getA(), getLastLayer()->getA()->getDataPointer(0), A, A->getDataPointer(0), getLastLayer()->getA()->getDataSize());
    }
    //计算误差
    if (error)
    {
        *error = getLastLayer()->calCostValue() / Y->getDataSize();
    }
}

void Net::moveDataPointer(int p)
{

}

//保存键结值，需配合ini中的网络结构
void Net::save(const std::string& filename)
{
    setDeviceSelf();
    if (filename == "") { return; }
    FILE* fout = stdout;
    LOG("Save net to %s...", filename.c_str());
    fout = fopen(filename.c_str(), "w+b");

    if (!fout)
    {
        LOG("Can not open file %s\n", filename.c_str());
        return;
    }

    for (int i_layer = 0; i_layer < getLayersCount(); i_layer++)
    {
        layer_vector_[i_layer]->save(fout);
    }
    fclose(fout);
    LOG("done\n");
}

//载入键结值，需配合ini中的网络结构
void Net::load(const std::string& filename)
{
    setDeviceSelf();
    if (filename == "") { return; }
    FILE* fin = stdout;
    LOG("Loading net from %s...", filename.c_str());
    fin = fopen(filename.c_str(), "r+b");
    if (!fin)
    {
        LOG("Can not open file %s\n", filename.c_str());
        return;
    }
    for (int i_layer = 0; i_layer < getLayersCount(); i_layer++)
    {
        layer_vector_[i_layer]->load(fin);
    }
    fclose(fin);
    LOG("done\n");
    trained_ = true;
}

//计算网络中参数的的L1和L2范数
void Net::calNorm(real& l1, real& l2)
{
    setDeviceSelf();
    l1 = 0, l2 = 0;
    //计算L1和L2
    for (auto l : layer_vector_)
    {
        if (l->getWeight())
        {
            l1 += l->getWeight()->sumAbs();
            l2 += l->getWeight()->dotSelf();
        }
        if (l->getBias())
        {
            l1 += l->getBias()->sumAbs();
            l2 += l->getBias()->dotSelf();
        }
    }
    //LOG("L1 = %g, L2 = %g\n", l1, l2);
}

void Net::malloc(int batch_size)
{
    //初始化
    int index = 0;
    for (auto& layer : layer_vector_)
    {
        LOG("---------- Layer %3d ----------\n", index);
        layer->setBatchSize(batch_size);
        layer->init();
        index++;
    }

    resetBatchSize(batch_);
    if (option_->getInt("LoadNet") != 0)
    {
        load(option_->getString("LoadFile"));
    }
}

//设置数据组数
int Net::resetBatchSize(int n)
{
    //对于一个网络，所有层数据组数应该一致
    if (n == layer_vector_[0]->getBatchSize()) { return n; }

    for (auto l : layer_vector_)
    {
        l->setBatchSize(n);
        l->resetGroupCount();
    }

    return n;
}

//调整整个网络的学习率
real Net::adjustLearnRate(int ec)
{
    real lr = 0;
    for (auto l : layer_vector_)
    {
        l->setEpochCount(ec);
        lr = l->adjustLearnRate();
    }
    return lr;
}

//返回值是max位置准确率，即标签正确率
real Net::test(const std::string& info, DataGroup& data, int force_output, int test_max)
{
    setDeviceSelf();

    if (getLayersCount() <= 0) { return -1; }

    int group_size = data.getNumber();
    if (group_size <= 0) { return 0; }

    setActivePhase(ACTIVE_PHASE_TEST);
    data.createA();

    DataGroup data_gpu;
    if (data.X()->getCudaType() == CUDA_GPU)
    {
        data_gpu.initWithReference(data, data.getNumber(), MATRIX_DATA_OUTSIDE, CUDA_GPU);
        data_gpu.shareData(data, 0);
    }
    else
    {
        data_gpu.cloneFrom(data);
    }
    data_gpu.createA();
    real error = 0;

    DataGroup data_sub;

    data_sub.initWithReference(data_gpu, batch_, MATRIX_DATA_OUTSIDE);

    for (int i = 0; i < group_size; i += batch_)
    {
        //检查最后一组是不是组数不足
        int n_rest = group_size - i;
        if (n_rest < batch_)
        {
            data_sub.resize(n_rest);
            resetBatchSize(n_rest);
        }
        data_sub.shareData(data_gpu, i);
        real e = 0;
        active(data_sub, false, false, &e);
        error += e * data_sub.getNumber();
    }
    error /= data_gpu.getNumber();
    Matrix::copyData(data_gpu.A(), data.A());

    //恢复网络原来的设置组数
    resetBatchSize(batch_);
    setActivePhase(ACTIVE_PHASE_TRAIN);

    int y_size = data.Y()->getRow();
    auto Y_cpu = data_gpu.Y()->clone(MATRIX_DATA_INSIDE, CUDA_CPU);
    auto A_cpu = data_gpu.A()->clone(MATRIX_DATA_INSIDE, CUDA_CPU);

    LOG("%s, %d groups of data\n", info.c_str(), group_size);
    if (error != 0) { LOG("Real error = %e\n", error); }
    for (int i = 0; i < std::min(group_size, force_output); i++)
    {
        for (int j = 0; j < y_size; j++)
        {
            LOG("%6.3f ", A_cpu->getData(j, i));
        }
        LOG(" --> ");
        for (int j = 0; j < y_size; j++)
        {
            LOG("%6.3f ", Y_cpu->getData(j, i));
        }
        LOG("\n");
    }
    real accuracy_total = 0;
    if (test_max)
    {
        auto A_max = new Matrix(A_cpu, MATRIX_DATA_INSIDE, CUDA_CPU);
        MatrixExtend::activeForward(ACTIVE_FUNCTION_FINDMAX, A_cpu, A_max);
        //A_max->print();

        for (int i = 0; i < std::min(group_size, force_output); i++)
        {
            int o = A_max->indexColMaxAbs(i);
            int e = data.Y()->indexColMaxAbs(i);
            LOG("%3d (%6.4f) --> %3d\n", o, A_cpu->getData(o, i), e);
        }
        std::vector<int> right(y_size), total(y_size);
        for (int j = 0; j < y_size; j++)
        {
            right[j] = 0;
            total[j] = 0;
        }
        int right_total = 0;

        for (int i = 0; i < group_size; i++)
        {
            for (int j = 0; j < y_size; j++)
            {
                if (Y_cpu->getData(j, i) == 1)
                {
                    total[j]++;
                    if (A_max->getData(j, i) == 1)
                    {
                        right[j]++;
                        right_total++;
                    }
                }
            }
        }
        safe_delete(A_max);
        accuracy_total = 1.0 * right_total / group_size;
        Color::set(CONSOLE_COLOR_LIGHT_RED);
        LOG("Total accuracy: %.2f%% (%d/%d) (error/total)\n", 100 * accuracy_total, group_size - right_total, group_size);

        for (int j = 0; j < y_size; j++)
        {
            double accur = 100.0 * right[j] / total[j];
            LOG("%d: %.2f%% (%d/%d), ", j, accur, total[j] - right[j], total[j]);
        }
        LOG("\b\b \n");

        Color::set(CONSOLE_COLOR_NONE);

    }
    safe_delete(A_cpu);
    safe_delete(Y_cpu);
    return accuracy_total;
}

void Net::setLearnRateBase(real lrb)
{
    for (auto& l : layer_vector_)
    {
        l->getSolver()->setLearnRateBase(lrb);
    }
}
