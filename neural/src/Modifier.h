#pragma once
#include "Matrix.h"
#include "Neural.h"
#include "Option.h"

//该类负责对weight进行部分微调，例如稀疏项可归于此类
class Modifier : public Neural
{
public:
    Modifier();
    virtual ~Modifier();

private:
    real sparse_beta_ = 0;
    real sparse_rou_ = 0.1;
    Matrix* sparse_rou_hat_ = nullptr;
    Matrix* sparse_rou_hat_vector_ = nullptr;
    Matrix* as_sparse_ = nullptr;
    int batch_ = 0;

    real diverse_beta_ = 0;
    real diverse_epsilon_;
    Matrix* diverse_aver_ = nullptr;
    Matrix* diverse_aver2_ = nullptr;
    Matrix* diverse_aver3_ = nullptr;
    Matrix* as_diverse_aver_ = nullptr;
    Matrix* diverse_workspace_ = nullptr;
    Matrix* diverse_A_ = nullptr;
    std::vector<int> diverse_workspace2_;
public:
    void init(Option* op, std::string section, Matrix* A);

private:
    void destory();

public:
    void modifyDA(Matrix*A, Matrix*dA);

};

