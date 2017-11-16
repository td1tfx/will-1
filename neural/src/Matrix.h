#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <cfloat>
#include <vector>
#include "types.h"
#include "CudaToolkit.h"
#include "Random.h"
#include "blas_types.h"
#include "cblas_real.h"


//数据位置（是否需要自己析构数据）
typedef enum
{
    MATRIX_DATA_OUTSIDE = 0,
    MATRIX_DATA_INSIDE,
} MatrixDataType;

//矩阵和张量
//该类更大的意义是封装cuda运算
//依赖cblas_real，cublas_real，cudnn_desc和cudaengine，其中cudaengine包装了一些cuda的必要参数
class Matrix
{
public:
    friend class MatrixExtend;
protected:
    //矩阵所使用的常数
    static const real const_real_1;
    static const real const_real_0;

    CudaToolkit* cuda_ = nullptr;

    CudaType cuda_type_ = CUDA_CPU;
    //CudaToolkit* cu_ = nullptr;

    real* data_ = nullptr;
    int row_ = 0;
    int col_ = 0;
    int64_t data_size_;
    MatrixDataType matrix_data_ = MATRIX_DATA_INSIDE;  //是否由矩阵自己保管数据，主要影响析构行为
    int64_t occupy_data_size_ = -1;  //实际占用的数据长度，当重设置尺寸不大于此值的时候不会重新分配内存

    //一列的数据作为一个或一组图像，矩阵本身是列优先
    //但是在图片处理，包含卷积核默认是行优先（遵从cudnn），也就是说图片和卷积核可以认为是转置保存的！！
    int width_, height_, channel_, number_;
    //int _row_, _col_, _width_, _height_, _channel_, _number_;

    cudnnTensorDescriptor_t tensor_desc_ = nullptr;

    //int nvml_id_;
public:
    Matrix(int m, int n, MatrixDataType try_inside = MATRIX_DATA_INSIDE, CudaType try_cuda = CUDA_GPU);
    Matrix(int w, int h, int c, int n, MatrixDataType try_inside = MATRIX_DATA_INSIDE, CudaType try_cuda = CUDA_GPU);
    Matrix(int w, int h, int c, int n, real* data, CudaType try_cuda = CUDA_GPU);
    Matrix(Matrix* src, MatrixDataType try_inside = MATRIX_DATA_INSIDE, CudaType try_cuda = CUDA_GPU);
    ~Matrix();
    Matrix* clone(MatrixDataType try_inside = MATRIX_DATA_INSIDE, CudaType try_cuda = CUDA_GPU);
    Matrix* cloneShared();
    Matrix* cloneSharedCol(int col = 1);
    CudaType getCudaType() { return cuda_type_; }
public:
    //注意有一些在矩阵模式下才能正确返回结果，有一些在4阶张量模式下使用，而实际上4阶张量模式的作用范围有限
    int mn2i(int m, int n) { return m + n * row_; }
    int whcn2i(int w, int h, int c, int n) { return w + h * width_ + c * width_ * height_ + n * channel_ * width_ * height_; }
    bool haveData(int w, int h, int c, int n) { int i = whcn2i(w, h, c, n); return (i >= 0 && i < data_size_); }

public:
    int getRow() { return row_; }
    int getCol() { return col_; }
    int getWidth() { return width_; }
    int getHeight() { return height_; }
    int getChannel() { return channel_; }
    int getNumber() { return number_; }
    int64_t getDataSize() { if (this == nullptr) { return 0; } return data_size_; }
    int64_t getDataSizeInByte() { return getDataSize() * sizeof(real); }

    //以下3个函数，注意如果数据在显存中，一般x来说是无法赋值和输出的
    real& getData(int i) { return data_[i]; }
    real& getData(int m, int n) { return data_[mn2i(m, n)]; }
    real& getData(int w, int h, int c, int n) { return data_[whcn2i(w, h, c, n)]; }

    real* getDataPointer() { return data_; }
    real* getDataPointer(int i) { return &getData(i); }
    real* getDataPointer(int r, int c) { return &getData(r, c); }
    real* getDataPointer(int w, int h, int c, int n) { return &getData(w, h, c, n); }
    real& operator [](int i) { return data_[i]; }
public:
    //改变矩阵维度，同时矩阵数据尺寸可能会变化，如果尺寸变大会备份数据
    int resize(int n, bool force = false);
    int resize(int m, int n, bool force = false);
    int resize(int w, int h, int c, int n, bool force = false);
    int resize(Matrix* X, bool force = false);

    //重设数据指针，这个函数可能不安全，慎用！！
    void resetDataPointer(real* d) { data_ = d; }

    //使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
    void setInsideData(MatrixDataType id) { matrix_data_ = id; }

    void print(FILE* fout = stdout);
    int save(FILE* fout = stdout);
    //int load(deque2<real>& v);
    int load(FILE* fin = stdout);
    void printAsVector(FILE* fout = stdout);
    //int loadAsVector(std::deque2<real>& v);
    void printAsMatrix(FILE* fout = stdout);

    void copyDataInFromHost(real* src, int64_t size);
    void copyDataOutToHost(real* dst, int64_t size);

    static void copyData(Matrix* A, Matrix* R, int64_t size = -1);
    static void copyDataPointer(Matrix* A, real* A_pointer, Matrix* R, real* R_pointer, int64_t size = -1);
    static void copyDataAcrossDevice(Matrix* A, Matrix* R, int64_t size = -1);

    void toGPU();
    void toCPU(bool reserve_data = true);
    void shareData(Matrix* A, int m = 0, int n = 0);
    void shareData(Matrix* A, int w, int h, int c, int n);
    void shareData(real* data);

    void filp(int flip_flag);
    void transpose(int transpose_flag);

    //静态运算函数在结果矩阵使用显存时就调用cuda函数计算，但是调用者应保证所有矩阵一致
    //在使用cuda的时候也有可能存在在内存中的矩阵

private:
    //一般来说，在X，A，dX，dA中，以X的计算设置为准，反向时一般不再重新设置
    //这些参量的初值均为空，一般来说，算法只设置一次，之后不重复设置

    //必须配对！
    real* mallocData(int64_t size);
    void freeData();

    //这两组好像必须交叉配对！
    real* mallocCPU_dataToCPU();
    void freeCPU(real* temp);
    real* mallocCPU();
    void dataToGPU_freeCPU(real* temp);

public:
    //运算函数
    void repeat(int c = 1);
    int indexColMaxAbs(int c);
    real sumAbs();
    real sumColAbs(int c);
    real sum();

    void initData(real v, int inc = 0);
    void initRandom(int seed = 0);
    void initRandom(Random<real>* r);
    void sectionLimit(real v0, real v1);
    void scale(real v);
    void scaleCol(real v, int c);

    static void mul(Matrix* A, Matrix* B, Matrix* R, real a = 1, real c = 0, MatrixTransType ta = MATRIX_NO_TRANS, MatrixTransType tb = MATRIX_NO_TRANS);
    static void mulVector(Matrix* A, Matrix* B, Matrix* R, real a = 1, real c = 0, MatrixTransType ta = MATRIX_NO_TRANS);
    static void mulVector2(Matrix* A, Matrix* B, Matrix* R, real a = 1, real c = 0, MatrixTransType ta = MATRIX_NO_TRANS);
    static void elementMul(Matrix* A, Matrix* B, Matrix* R, real a = 1, real b = 0);
    //static void mul(Matrix* A, Matrix* B, Matrix* R, real a = 1, real b = 1);
    static void add(Matrix* A, Matrix* B, Matrix* R, real a = 1, real b = 1);
    static real dot(Matrix* A, Matrix* B);
    static real dotCol(Matrix* A, int cA, Matrix* B, int cB);
    static real dotPart(int size, Matrix* A, real* a, int cA, real* b, int cB);
    real dotSelf();
    static void sign(Matrix* A, Matrix* R, real v = 1, real section = 1e-4);
public:

    //PYTHON only ----------------------------------------------------------------------------------------------------
    //取值和赋值，通常不推荐在c++中使用，仅用于python接口，故安全保护较多
    real getDataValue(int i)
    {
        if (cuda_type_ == CUDA_CPU && i >= 0 && i < data_size_)
        {
            return getData(i);
        }
        return 0;
    }
    real getDataValue(int m, int n) { return getDataValue(mn2i(m, n)); }
    real getDataValue(int w, int h, int c, int n) { return getDataValue(whcn2i(w, h, c, n)); }

    void setDataValue(float v, int i)
    {
        if (cuda_type_ == CUDA_CPU && i >= 0 && i < data_size_)
        {
            getData(i) = v;
        }
    }
    void setDataValue(float v, int m, int n) { setDataValue(v, mn2i(m, n)); }
    void setDataValue(float v, int w, int h, int c, int n) { setDataValue(v, whcn2i(w, h, c, n)); }

    void importData(real* v, int n);
    void exportData(real* v, int n);

    //PYTHON only ----------------------------------------------------------------------------------------------------

public:

    //以下函数都是自己写cuda部分
    void reciprocal(real scale = 1);
    void addNumber(real v, real scale = 1);
    static void elementPow(Matrix* A, Matrix* R, real e, real bias = 0);
    static void elementDiv(Matrix* A, Matrix* B, Matrix* R, real a = 0, real b = 0);

    // the function is priviate for concate the data and append the data
    static void concatByChannel(std::vector<Matrix*> A_vector, Matrix* R);
    static void splitByChannel(Matrix* A, std::vector<Matrix*> R_vector);

};

