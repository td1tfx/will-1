#pragma once
#include "Matrix.h"

//该类中均不是矩阵基本计算，全部为静态函数
class MatrixExtend : public Matrix
{
private:
    //MatrixExtend();
    virtual ~MatrixExtend();

public:
    //以下函数不属于矩阵基本运算

    //激活和反向激活中，输入和输出矩阵都是同维度
    static void activeForward(ActiveFunctionType af, Matrix* X, Matrix* A);
    static void activeBackward(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX);

    static void activeForwardEx(ActiveFunctionType af, Matrix* X, Matrix* A,
        std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix*>& matrix_vector);
    static void activeBackwardEx(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
        std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix*>& matrix_vector);

    //正向统一由X以及其他参数生成A，前两个参数必定是X，A；反向由A，DA，X生成DX以及其他参数（例如dW），前4个参数必定是A，DA，X，DX
    static void poolingForward(Matrix* X, Matrix* A, PoolingType pooling_type,
        int window_w, int window_h, int stride_w, int stride_h, int padding_w = 0, int padding_h = 0, real a = 1, real b = 0);
    static void poolingBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, PoolingType pooling_type,
        int window_w, int window_h, int stride_w, int stride_h, int padding_w = 0, int padding_h = 0, real a = 1, real b = 0);

    static void convolutionForward(Matrix* X, Matrix* A, Matrix* W, Matrix* workspace, std::vector<int>& workspace_forward,
        int stride_w = 1, int stride_h = 1, int padding_w = 0, int padding_h = 0, real a = 1, real b = 0);
    static void convolutionBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* W, Matrix* dW, Matrix* dB, Matrix* workspace,
        std::vector<int>& workspace_backward_dx, std::vector<int>& workspace_backward_dw,
        int stride_w = 1, int stride_h = 1, int padding_w = 0, int padding_h = 0, real a = 1, real b = 0);

    static void adaUpdate(Matrix* E_dw2, Matrix* E_g2, Matrix* dw, real rou, real epsilon);
    static void adaDeltaUpdate(Matrix* E_g2, Matrix* E_d2, Matrix* g, Matrix* d, real rou, real epsilon);

};

