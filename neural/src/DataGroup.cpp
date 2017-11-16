#include "DataGroup.h"

DataGroup::DataGroup()
{
}

DataGroup::~DataGroup()
{
    clear();
}

void DataGroup::clear()
{
    safe_delete({ &X_, &Y_, &A_ });
}

void DataGroup::initWithReference(const DataGroup& data, int number, MatrixDataType try_inside /*= MATRIX_DATA_INSIDE*/, CudaType try_cuda /*= CUDA_GPU*/)
{
    clear();
    if (data.X_)
    {
        X_ = new Matrix(data.X_->getWidth(), data.X_->getHeight(), data.X_->getChannel(), number, try_inside, try_cuda);
    }
    if (data.Y_)
    {
        Y_ = new Matrix(data.Y_->getWidth(), data.Y_->getHeight(), data.Y_->getChannel(), number, try_inside, try_cuda);
    }
    if (data.A_)
    {
        A_ = new Matrix(data.A_->getWidth(), data.A_->getHeight(), data.A_->getChannel(), number, try_inside, try_cuda);
    }
}

void DataGroup::cloneFrom(const DataGroup& data, MatrixDataType try_inside /* = MATRIX_DATA_INSIDE*/, CudaType try_cuda /*= CUDA_GPU*/)
{
    clear();
    if (data.X_) { X_ = data.X_->clone(try_inside, try_cuda); }
    if (data.Y_) { Y_ = data.Y_->clone(try_inside, try_cuda); }
    if (data.A_) { A_ = data.Y_->clone(try_inside, try_cuda); }
}

void DataGroup::createA(MatrixDataType try_inside /*= MATRIX_DATA_INSIDE*/)
{
    if (Y_ && A_ == nullptr)
    {
        A_ = new Matrix(Y_, try_inside, Y_->getCudaType());
    }
}

void DataGroup::resize(int n)
{
    if (X_) { X_->resize(n); }
    if (Y_) { Y_->resize(n); }
    if (A_) { A_->resize(n); }
}

void DataGroup::resize(const DataGroup& data, int n)
{
    if (X_) { X_->resize(n); }
    if (Y_) { Y_->resize(n); }
    if (A_) { A_->resize(n); }
}

void DataGroup::shareData(const DataGroup& data, int n)
{
    if (X_) { X_->shareData(data.X_, 0, n); }
    if (Y_) { Y_->shareData(data.Y_, 0, n); }
    if (A_) { A_->shareData(data.A_, 0, n); }
}

void DataGroup::copyPartFrom(const DataGroup& data, int p0, int p1, int n, bool needA/* = false*/)
{
    if (X_) { Matrix::copyDataPointer(data.X_, data.X_->getDataPointer(0, p0), X_, X_->getDataPointer(0, p1), X_->getRow() * n); }
    if (Y_) { Matrix::copyDataPointer(data.Y_, data.Y_->getDataPointer(0, p0), Y_, Y_->getDataPointer(0, p1), Y_->getRow() * n); }
    if (needA && A_) { Matrix::copyDataPointer(data.A_, data.A_->getDataPointer(0, p0), A_, A_->getDataPointer(0, p1), A_->getRow() * n); }
}

void DataGroup::toGPU()
{
    if (X_) { X_->toGPU(); }
    if (Y_) { Y_->toGPU(); }
    if (A_) { A_->toGPU(); }
}

void DataGroup::toCPU()
{
    if (X_) { X_->toCPU(); }
    if (Y_) { Y_->toCPU(); }
    if (A_) { A_->toCPU(); }
}
