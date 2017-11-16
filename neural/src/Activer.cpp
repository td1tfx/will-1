#include "Activer.h"

Activer::Activer()
{
}


Activer::~Activer()
{
    safe_delete(active_matrix_vector_);
}

void Activer::init(Option* op, std::string section, LayerConnectionType ct)
{
    active_function_ = op->getActiveFunctionTypeFromSection(section, "active", "none");
    real learn_rate_base = op->getRealFromSection2(section, "learn_rate_base", 1e-2);
}

//激活前的准备工作，主要用于一些激活函数在训练和测试时有不同设置
void Activer::activePrepare(ActivePhaseType ap)
{

}

void Activer::initActiveMatrixVector(int n)
{
    for (auto& m : active_matrix_vector_)
    {
        delete m;
    }
    active_matrix_vector_.resize(n);
    for (auto& m : active_matrix_vector_)
    {
        m = new Matrix(0, 0);
    }
}

//dY will be changed
real Activer::calCostValue(Matrix* A, Matrix* dA, Matrix* Y, Matrix* dY)
{
    if (!dY) { return 0; }
    switch (active_function_)
    {
    case ACTIVE_FUNCTION_SOFTMAX:
        //Matrix::activeForward(ActiveFunction_Softmax_Log, XMatrix, dYMatrix);
        Matrix::elementMul(dA, Y, dY);
        break;
    default:
        //平方和误差
        Matrix::elementMul(dA, dA, dY);
        break;
    }
    //return weight_parameter->dotSelf();
    return dY->sumAbs();
}

void Activer::save(FILE* fout)
{
    if (active_function_ == ACTIVE_FUNCTION_BATCH_NORMALIZATION || active_function_ == ACTIVE_FUNCTION_SPATIAL_TRANSFORMER)
    {
        for (auto ex : active_matrix_vector_)
        {
            ex->save(fout);
        }
    }
}

void Activer::load(FILE* fin)
{
    if (active_function_ == ACTIVE_FUNCTION_BATCH_NORMALIZATION || active_function_ == ACTIVE_FUNCTION_SPATIAL_TRANSFORMER)
    {
        for (auto ex : active_matrix_vector_)
        {
            ex->load(fin);
        }
    }
}
