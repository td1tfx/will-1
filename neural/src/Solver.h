#pragma once
#include "Neural.h"
#include "Matrix.h"
#include "Option.h"

//更新参数的策略
class Solver : public Neural
{
public:
    Solver();
    virtual ~Solver();

protected:
    real momentum_ = 0;    //上次的dWeight保留多少

    //学习率调整相关
    real learn_rate_base_; //基础学习率
    real learn_rate_;      //学习率
    real weight_decay_, weight_decay_l1_;    //正则化参数，防止过拟合
    real derror0_ = 1;     //暂时无用

    Matrix* W_sign_ = nullptr;

    real lr_gamma_ = 0.0001;
    real lr_power_ = 0.75;

    real lr_weight_scale_ = 1;
    real lr_bias_scale_ = 2;

    int lr_step_;
    std::vector<real> lr_step_rate_;

    real nag_momentum_ = 0;
    Matrix* nag_w_ = nullptr;
    Matrix* nag_b_ = nullptr;

    real ada_epsilon_ = 1e-6;
    real ada_rou_ = 0.95;
    Matrix* ada_mean_gw2_ = nullptr;
    Matrix* ada_mean_dw2_ = nullptr;
    Matrix* ada_step_w_ = nullptr;
    Matrix* ada_mean_gb2_ = nullptr;
    Matrix* ada_mean_db2_ = nullptr;
    Matrix* ada_step_b_ = nullptr;

    SolverType solver_ = SOLVER_SGD;
    AdjustLearnRateType lr_adjust_method_ = ADJUST_LEARN_RATE_FIXED;

    bool normalized_dweight_ = false;

    int batch_;

public:
    real getMomentum() { return momentum_; }
    void setLearnRateBase(real lrb) { learn_rate_base_ = lrb; }
    real getLearnRateBase() { return learn_rate_base_; }
    real getLearnRate() { return learn_rate_; }

public:
    void init(Option* op, std::string section, int row, int batch, Matrix* W, Matrix* b);
    real adjustLearnRate(int epoch);
    void updateParametersPre(Matrix* W, Matrix* dW, Matrix* b, Matrix* db);
    void updateParameters(Matrix* W, Matrix* dW, Matrix* b, Matrix* db);
private:
    void destory();
};

