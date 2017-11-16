#include "Layer.h"
#include "Random.h"
#include "MatrixFiller.h"
#include "MatrixExtend.h"

Layer::Layer()
{
    activer_ = new Activer();
    solver_ = new Solver();
    modifier_ = new Modifier();
}

Layer::~Layer()
{
    destroyData();
    safe_delete(activer_);
    safe_delete(solver_);
    safe_delete(modifier_);
    //fprintf(stderr, "~Layer.\n");
}

void Layer::destroyData()
{
    safe_delete({ &X_, &dX_, &A_, &dA_, &Y_, &dY_ });
    safe_delete({ &W_, &dW_, &b_, &db_ });
}

void Layer::resetGroupCount()
{
    for (auto& m : { X_, dX_, A_, dA_, Y_, dY_ })
    {
        if (m) { m->resize(batch_); }
    }
    resetGroupCount2();
}

//设置名字，顺便设置输入和输出性质
void Layer::setName(const std::string& name)
{
    layer_name_ = name;
    visible_type_ = LAYER_VISIBLE_HIDDEN;
}

//这个会检查前一层
LayerConnectionType Layer::getConnection2()
{
    auto ct = connetion_type_;
    if (connetion_type_ == LAYER_CONNECTION_DIRECT) { ct = prev_layer_->connetion_type_; }
    return ct;
}

//必须先设置option和layer_name
void Layer::init()
{
    //公共部分，各层的学习速度和正则化系数
    //先获取全局的，自己块的如果有会覆盖全局

    activer_->init(option_, layer_name_, getConnection2());

    need_bias_ = option_->getIntFromSection(layer_name_, "need_bias", 1) != 0;
    need_train_ = option_->getIntFromSection(layer_name_, "need_train", 1) != 0;

    //不同类层分别设置，子类中不处理A！
    init2();

    //在无激活的情况，A与X同
    if (activer_->none())
    {
        A_ = X_->cloneShared();
        dA_ = dX_->cloneShared();
        dA_->initData(0);
    }
    else
    {
        A_ = new Matrix(X_);
        dA_ = new Matrix(X_);
        A_->initData(0);
        dA_->initData(0);
    }
    //输出层多出一个Y设置
    if (visible_type_ == LAYER_VISIBLE_OUT)
    {
        Y_ = new Matrix(A_, MATRIX_DATA_OUTSIDE);
        dY_ = new Matrix(Y_);
    }

    //求解器相关
    solver_->init(option_, layer_name_, out_total_, batch_, W_, b_);
    //稀疏项相关
    modifier_->init(option_, layer_name_, A_);

    //输出本层信息
    LOG("  name: %s\n", layer_name_.c_str());
    LOG("  type: %s\n", option_->getStringFromSection(layer_name_, "type").c_str());
    LOG("  out nodes: %d\n", out_total_);
    LOG("  width, height, channel: %d, %d, %d\n", out_width_, out_height_, out_channel_);
    if (W_)
    {
        LOG("  weight and bias size: %lld, %lld\n", W_->getDataSize(), b_->getDataSize());
    }
    LOG("  x data size: %d, %d\n", X_->getRow(), X_->getCol());
    //LOG("  learn rate = %g, weight decay = %g, momentum = %g, gamma = %g, power = %g\n",
    //learn_rate_base_, weight_decay_, momentum_, lr_gamma_, lr_power_);
    LOG("  have bias: %d\n", need_bias_);
    //LOG("  solver: %s\n", option_->getStringFromSection2(layer_name_, "solver", "sgd").c_str());
    if (!next_layers_.empty())
    {
        LOG("  next layer(s): ");
        for (auto& l : next_layers_)
        {
            LOG("%s, ", l->layer_name_.c_str());
        }
        LOG("\b\b \n");
    }
}

//以下为前向和后向的计算函数
//计算损失函数的值，通常在训练过程中没必要显式计算
real Layer::calCostValue()
{
    return activer_->calCostValue(A_, dA_, Y_, dY_);
}

//calCostDA和DX是最后一层的反向回传，最终的目的其实是计算DX
//事实上最后一层是否计算DA并不重要，下面计算DA的方法其实并不正确
void Layer::calCostDA()
{
    //partial C / partial A
    Matrix::add(A_, Y_, dA_, 1, -1);
}

void Layer::calCostDX()
{
    Matrix::copyData(dA_, dX_);
}

void Layer::activeForward()
{
    updateX();
    updateA();
    if (visible_type_ == LAYER_VISIBLE_OUT)
    {
        calCostDA();
    }
    //LOG("%s,%g,%g,%g\n", layer_name_.c_str(), X_->dotSelf(), A_->dotSelf(), prev_layer_->A_->dotSelf());
    //if (W_) { LOG("%s,%g,%g\n", layer_name_.c_str(), W_->dotSelf(), b_->dotSelf()); }
}

//这里实际只包含了作为输出层的实现，即代价函数的形式，其他层交给各自的子类
void Layer::activeBackward()
{
    updateDA();
    updateDX();
    //updateDParameters(solver_->getMomentum());
    //updateParameters();    //看起来这个应该放在外面的样子
}

void Layer::updateABackward()
{
    Matrix::add(A_, dA_, A_, 1, -solver_->getLearnRateBase());
}

void Layer::updateA()
{
    activer_->forward(X_, A_);
}

//更新dA
void Layer::updateDA()
{
    //实际上代价函数的导数形式不必拘泥于具体的推导，用dA就够了
    if (this->visible_type_ != LAYER_VISIBLE_OUT)
    {
        dA_->initData(0);
        for (auto& l : next_layers_)
        {
            l->updatePrevLayerDA();
        }
    }
    modifier_->modifyDA(A_, dA_);
}

//更新dX
void Layer::updateDX()
{
    if (visible_type_ == LAYER_VISIBLE_OUT)
    {
        calCostDX();
    }
    else
    {
        activer_->backward(A_, dA_, X_, dX_);
    }
}

//在计算参数的变化率之前先行更新参数，NAG使用
void Layer::updateParametersPre()
{
    solver_->updateParametersPre(W_, dW_, b_, db_);
}

//更新参数
void Layer::updateParameters()
{
    updateDParameters(solver_->getMomentum());
    solver_->updateParameters(W_, dW_, b_, db_);
}

void Layer::save(FILE* fout)
{
    if (option_->getInt("save_all_layers", 1) || option_->getIntFromSection(layer_name_, "save", 0))
    {
        if (W_) { W_->save(fout); }
        if (b_) { b_->save(fout); }
        activer_->save(fout);
    }
}

void Layer::load(FILE* fin)
{
    //随便激活一次，初始化部分值
    updateA();
    updateDX();
    if (option_->getInt("load_all_layers", 1) || option_->getIntFromSection(layer_name_, "load", 0))
    {
        if (W_) { W_->load(fin); }
        if (b_) { b_->load(fin); }
        activer_->load(fin);
    }
}
