#pragma once
#include "Neural.h"
#include "Matrix.h"
#include "Option.h"
#include "MatrixExtend.h"
#include "Random.h"

class Activer : public Neural
{
public:
    Activer();
    virtual ~Activer();

protected:
    //激活函数相关
    ActiveFunctionType active_function_ = ACTIVE_FUNCTION_NONE;
    //bool need_active_ex_ = false;
    std::vector<real> active_real_vector_;
    std::vector<int> active_int_vector_;
    std::vector<Matrix*> active_matrix_vector_;
    //代价函数，其实没啥用
    CostFunctionType cost_function_ = COST_FUNCTION_CROSS_ENTROPY;

    Random<real> random_generator_;
    ActivePhaseType active_phase_ = ACTIVE_PHASE_TRAIN;

public:

    void forward(Matrix* X, Matrix* A)
    {
        activePrepare(active_phase_);
        MatrixExtend::activeForwardEx(active_function_, X, A, active_int_vector_, active_real_vector_, active_matrix_vector_);
    }

    void backward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX)
    {
        MatrixExtend::activeBackwardEx(active_function_, A, dA, X, dX, active_int_vector_, active_real_vector_, active_matrix_vector_);
    }

    bool none() { return active_function_ == ACTIVE_FUNCTION_NONE; }
    void init(Option* op, std::string section, LayerConnectionType ct);
    void activePrepare(ActivePhaseType ap);
    void initActiveMatrixVector(int n);
    void setCostFunction(CostFunctionType cf) { cost_function_ = cf; }

    real calCostValue(Matrix* A, Matrix* dA, Matrix* Y, Matrix* dY);

    ActiveFunctionType getActiveFunction() { return active_function_; }
    void setActivePhase(ActivePhaseType ap) { active_phase_ = ap; }

public:
    void save(FILE* fout);
    void load(FILE* fin);


};

