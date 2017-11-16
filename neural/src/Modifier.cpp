#include "Modifier.h"
#include "MatrixExtend.h"

Modifier::Modifier()
{
}


Modifier::~Modifier()
{
    destory();
}

void Modifier::init(Option* op, std::string section, Matrix* A)
{
    batch_ = A->getNumber();

    sparse_beta_ = op->getRealFromSection2(section, "sparse_beta", 0);
    if (sparse_beta_ != 0)
    {
        sparse_rou_ = op->getRealFromSection2(section, "sparse_rou", 0.1);
        sparse_rou_hat_ = new Matrix(A);
        sparse_rou_hat_vector_ = new Matrix(A->getRow(), 1);
        as_sparse_ = new Matrix(batch_, 1);
        as_sparse_->initData(1);
    }

    diverse_beta_ = op->getRealFromSection2(section, "diverse_beta", 0);
    if (diverse_beta_ != 0)
    {
        diverse_epsilon_ = op->getRealFromSection2(section, "diverse_epsilon", 1e-8);
        diverse_aver_ = new Matrix(A);
        diverse_aver2_ = new Matrix(A);
        diverse_aver3_ = new Matrix(A);
        as_diverse_aver_ = new Matrix(1, 1, A->getChannel() * A->getNumber(), 1);
        as_diverse_aver_->initData(1);
        diverse_A_ = A->cloneShared();
        diverse_A_->resize(1, 1, A->getChannel() * A->getNumber(), 1);
        diverse_workspace_ = new Matrix(1, int(1e6));
    }
}

void Modifier::destory()
{
    safe_delete({ &sparse_rou_hat_, &sparse_rou_hat_vector_, &as_sparse_ });
    safe_delete({ &diverse_aver_, &diverse_aver2_, &diverse_aver3_, &as_diverse_aver_, &diverse_workspace_, &diverse_A_ });
}

//the extra-cost function often modifies dA with A
void Modifier::modifyDA(Matrix* A, Matrix* dA)
{

}

