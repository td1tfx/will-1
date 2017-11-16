#include "Solver.h"
#include "MatrixExtend.h"

Solver::Solver()
{
}


Solver::~Solver()
{
    destory();
}

void Solver::destory()
{
    safe_delete({ &W_sign_ });
    safe_delete({ &ada_mean_gw2_, &ada_mean_dw2_, &ada_step_w_, &ada_mean_gb2_, &ada_mean_db2_, &ada_step_b_ });
}

//求解器即如何更新参数的定义
void Solver::init(Option* op, std::string section, int row, int batch, Matrix* W, Matrix* b)
{
    batch_ = batch;
    learn_rate_base_ = op->getRealFromSection2(section, "learn_rate_base", 1e-2);
    learn_rate_ = learn_rate_base_;
    weight_decay_ = op->getRealFromSection2(section, "weight_decay", 0);
    weight_decay_l1_ = op->getRealFromSection2(section, "weight_decay_l1", 0);

    normalized_dweight_ = op->getIntFromSection2(section, "normalized_ddeight", 0) != 0;

    if (weight_decay_l1_ != 0)
    {
        W_sign_ = new Matrix(W);
    }

    //学习率调整方案相关参数
    lr_adjust_method_ = op->getAdjustLearnRateType("lr_adjust_method", "fixed");
    lr_gamma_ = op->getRealFromSection2(section, "lr_gamma", 1e-4);
    lr_power_ = op->getRealFromSection2(section, "lr_power", 0.75);
    lr_weight_scale_ = op->getRealFromSection2(section, "lr_weight_scale", 1);
    lr_bias_scale_ = op->getRealFromSection2(section, "lr_bias_scale", 2);
    lr_step_ = std::max(1, int(op->getRealFromSection2(section, "lr_step", 1)));
    convert::findNumbers(op->getStringFromSection2(section, "lr_step_rate", "1"), &lr_step_rate_);

    //求解器设定
    auto solver_string = op->getStringFromSection2(section, "solver", "sgd");
    solver_ = op->getSolverType(solver_string);
    switch (solver_)
    {
    case SOLVER_SGD:
        momentum_ = op->getRealFromSection2(section, "momentum", 0.9);
        break;
    case SOLVER_NAG:
        nag_momentum_ = op->getRealFromSection2(section, "momentum", 0.9);
        momentum_ = 0;
        if (W)
        {
            nag_w_ = W->clone();
        }
        if (b)
        {
            nag_b_ = b->clone();
        }
        break;
    case SOLVER_ADA_DELTA:
        momentum_ = 0;
        ada_epsilon_ = op->getRealFromSection2(section, "ada_epsilon", 1e-6);
        ada_rou_ = op->getRealFromSection2(section, "ada_rou", 0.95);
        if (W)
        {
            ada_mean_gw2_ = new Matrix(W); // w's accumulation gradient
            ada_mean_dw2_ = new Matrix(W); // w's accumulation updates
            ada_step_w_ = new Matrix(W);
            ada_mean_gw2_->initData(0);    // initilize as 0, it is right
            ada_mean_dw2_->initData(0);
            ada_step_w_->initData(0);
        }
        if (b)
        {
            ada_mean_gb2_ = new Matrix(b); // b's accumulation gradient
            ada_mean_db2_ = new Matrix(b); // b's accumulation updates
            ada_step_b_ = new Matrix(b);
            ada_mean_gb2_->initData(0);
            ada_mean_db2_->initData(0);
            ada_step_b_->initData(0);
        }
        //learn_rate_ = 1;
        //lr_adjust_method_ = ADJUST_LEARN_RATE_FIXED;
        break;
    }
}

//以下求解器相关
//调整学习率
real Solver::adjustLearnRate(int epoch)
{
    switch (lr_adjust_method_)
    {
    case ADJUST_LEARN_RATE_FIXED:
        learn_rate_ = learn_rate_base_;
        break;
    case ADJUST_LEARN_RATE_INV:
        learn_rate_ = learn_rate_base_ * pow(1 + lr_power_ * epoch, -lr_power_);
        break;
    case ADJUST_LEARN_RATE_STEP:
    {
        int step_index = lr_step_rate_.size() - 1;
        step_index = std::min(epoch / lr_step_, step_index);
        learn_rate_ = learn_rate_base_ * lr_step_rate_[step_index];
        break;
    }
    default:
        break;
    }
    return learn_rate_;
}

void Solver::updateParametersPre(Matrix* W, Matrix* dW, Matrix* b, Matrix* db)
{
    if (W)
    {
        switch (solver_)
        {
        case SOLVER_NAG:
            //依据上一次的参数直接跳一步
            Matrix::add(W, nag_w_, W, 1 + nag_momentum_, -nag_momentum_);
            if (b) { Matrix::add(b, nag_b_, b, 1 + nag_momentum_, -nag_momentum_); }
            break;
        }
    }
}

//求解器本身（可以独立并行）
void Solver::updateParameters(Matrix* W, Matrix* dW, Matrix* b, Matrix* db)
{
    if (W)
    {
        switch (solver_)
        {
        case SOLVER_SGD:
        case SOLVER_NAG:
            if (normalized_dweight_)
            {
                dW->scale(sqrt(1.0 / dW->dotSelf()));
                if (db)
                {
                    db->scale(sqrt(1.0 / db->dotSelf()));
                }
            }
            Matrix::add(W, dW, W, 1 - weight_decay_ * learn_rate_, -learn_rate_ * lr_weight_scale_ / batch_);
            if (weight_decay_l1_ != 0)
            {
                Matrix::sign(W, W_sign_, weight_decay_l1_ * learn_rate_);
                Matrix::add(W, W_sign_, W, 1, -1);
            }
            if (b) { Matrix::add(b, db, b, 1, -learn_rate_ * lr_bias_scale_ / batch_); }

            if (solver_ == SOLVER_NAG)
            {
                Matrix::copyData(W, nag_w_);
                if (b) { Matrix::copyData(b, nag_b_); }
            }
            break;
        case SOLVER_ADA_DELTA:
            //LOG("ADADELTA\n");
            dW->scale(1.0 / batch_);
            MatrixExtend::adaDeltaUpdate(ada_mean_gw2_, ada_mean_dw2_, dW, ada_step_w_, ada_rou_, ada_epsilon_);
            Matrix::add(W, ada_step_w_, W, 1, -1);
            if (b)
            {
                db->scale(1.0 / batch_);
                MatrixExtend::adaDeltaUpdate(ada_mean_gb2_, ada_mean_db2_, db, ada_step_b_, ada_rou_, ada_epsilon_);
                Matrix::add(b, ada_step_b_, b, 1, -1);
            }
            break;
        }
    }
}
