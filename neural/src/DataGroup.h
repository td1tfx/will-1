#pragma once
#include "Matrix.h"
#include "Neural.h"

//this class is used to simple codes, not been packaged strictly
//use the pointers carefully to avoid wild pointer

class DataGroup : public Neural
{
public:
    DataGroup();
    virtual ~DataGroup();

private:
    Matrix* X_ = nullptr;
    Matrix* Y_ = nullptr;
    Matrix* A_ = nullptr;

public:
    Matrix* X() { return X_; }
    Matrix* Y() { return Y_; }
    Matrix* A() { return A_; }
    void setX(Matrix* x) { safe_delete(X_); X_ = x; }
    void setY(Matrix* y) { safe_delete(Y_); Y_ = y; }
    void setA(Matrix* a) { safe_delete(A_); A_ = a; }

    void clear();
    void initWithReference(const DataGroup& data, int number, MatrixDataType try_inside = MATRIX_DATA_INSIDE, CudaType try_cuda = CUDA_GPU);
    void cloneFrom(const DataGroup& data, MatrixDataType try_inside = MATRIX_DATA_INSIDE, CudaType try_cuda = CUDA_GPU);
    void createA(MatrixDataType try_inside = MATRIX_DATA_INSIDE);

    void resize(int n);
    void resize(const DataGroup& data, int n);

    void shareData(const DataGroup& data, int n);
    bool exist() { return X_ != nullptr; }
    int getNumber() { if (X_ == nullptr) { return 0; } return X_->getNumber(); }
    void copyPartFrom(const DataGroup& data, int p0, int p1, int n, bool needA = false);

    void toGPU();
    void toCPU();
};

