#include "MatrixExtend.h"
#include "VectorMath.h"


MatrixExtend::~MatrixExtend()
{
}


//正向激活，依据X计算A
//此处我们定义激活操作为输入和输出矩阵（或张量）的维度完全相同
void MatrixExtend::activeForward(ActiveFunctionType af, Matrix* X, Matrix* A)
{
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = X->cuda_;
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
        if (A->matrix_data_ == MATRIX_DATA_INSIDE)
        {
            Matrix::copyData(X, A);
        }
        else
        {
            A->shareData(X->getDataPointer());
        }
        break;
    case ACTIVE_FUNCTION_SIGMOID:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &const_real_1, X->tensor_desc_, X->data_, &const_real_0, A->tensor_desc_, A->data_);
        }
        else
        {
            VectorMath::sigmoid_v(X->data_, A->data_, A->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &const_real_1, X->tensor_desc_, X->data_, &const_real_0, A->tensor_desc_, A->data_);
        }
        else
        {
            VectorMath::relu_v(X->data_, A->data_, A->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &const_real_1, X->tensor_desc_, X->data_, &const_real_0, A->tensor_desc_, A->data_);
        }
        else
        {
            VectorMath::tanh_v(X->data_, A->data_, A->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setTensorDesc(cuda->tensor_desc_, 1, 1, X->row_, X->col_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                &const_real_1, cuda->tensor_desc_, X->data_, &const_real_0, cuda->tensor_desc_, A->data_);
        }
        else
        {
            //因为数值问题，可能需要减去每列最大值
            MatrixExtend::copyData(X, A);
            for (int i = 0; i < A->col_; i++)
            {
                VectorMath::minus_max(A->getDataPointer(0, i), A->row_);
            }
            VectorMath::exp_v(A->data_, A->data_, A->data_size_);
            for (int i = 0; i < A->col_; i++)
            {
                real sum = A->sumColAbs(i);
                if (sum == 0) { continue; }
                A->scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setTensorDesc(cuda->tensor_desc_, 1, 1, X->row_, X->col_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
                &const_real_1, cuda->tensor_desc_, X->data_, &const_real_0, cuda->tensor_desc_, A->data_);
        }
        else
        {
            VectorMath::exp_v(X->data_, A->data_, A->data_size_);
            for (int i = 0; i < A->col_; i++)
            {
                real sum = A->sumColAbs(i);
                if (sum == 0) { continue; }
                A->scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setTensorDesc(cuda->tensor_desc_, 1, 1, X->row_, X->col_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &const_real_1, cuda->tensor_desc_, X->data_, &const_real_0, cuda->tensor_desc_, A->data_);
        }
        else
        {
            activeForward(ACTIVE_FUNCTION_SOFTMAX, X, A);
            VectorMath::log_v(A->data_, A->data_, A->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_FINDMAX:
        //计算时尽量不要使用，只用在验证时
        if (A->data_size_ <= 0) { return; }
        if (X->cuda_type_ == CUDA_GPU)
        {
            auto T = new Matrix(A->row_, A->col_, MATRIX_DATA_INSIDE, CUDA_CPU);
            T->initData(0);
            for (int i_group = 0; i_group < A->col_; i_group++)
            {
                int index = X->indexColMaxAbs(i_group);
                T->getData(index, i_group) = 1;
            }
            copyData(T, A);
            delete T;
        }
        else
        {
            A->initData(0);
            for (int i_group = 0; i_group < A->col_; i_group++)
            {
                int index = X->indexColMaxAbs(i_group);
                A->getData(index, i_group) = 1;
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTPLUS:
        //GPU部分不支持
        if (X->cuda_type_ == CUDA_GPU)
        {
            fprintf(stderr, "Unsupported softplus on GPU!\n");
        }
        else
        {
            VectorMath::softplus_v(X->data_, A->data_, A->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_SQUARE:
        Matrix::elementMul(X, X, A);
        break;
    default:
        fprintf(stderr, "Parameters not enough!\n");
        break;
    }
}

//反向激活，依据X，A，dA计算dX
void MatrixExtend::activeBackward(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX)
{
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = dA->cuda_;
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
        if (dX->matrix_data_ == MATRIX_DATA_INSIDE)
        {
            Matrix::copyData(dA, dX);
        }
        else
        {
            dX->shareData(dA->getDataPointer());
        }
        break;
    case ACTIVE_FUNCTION_SIGMOID:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &const_real_1, A->tensor_desc_, A->data_,
                dA->tensor_desc_, dA->data_, X->tensor_desc_, X->data_, &const_real_0, dX->tensor_desc_, dX->data_);
        }
        else
        {
            VectorMath::sigmoid_vb(A->data_, dA->data_, X->data_, dX->data_, dX->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &const_real_1, A->tensor_desc_, A->data_,
                dA->tensor_desc_, dA->data_, X->tensor_desc_, X->data_, &const_real_0, dX->tensor_desc_, dX->data_);
        }
        else
        {
            VectorMath::relu_vb(A->data_, dA->data_, X->data_, dX->data_, dX->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        //两者结果在1e-10的精度有区别
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &const_real_1, A->tensor_desc_, A->data_,
                dA->tensor_desc_, dA->data_, X->tensor_desc_, X->data_, &const_real_0, dX->tensor_desc_, dX->data_);
        }
        else
        {
            VectorMath::tanh_vb(A->data_, dA->data_, X->data_, dX->data_, dX->data_size_);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
        if (X->cuda_type_ == CUDA_GPU)
        {
            auto softmax_flag = CUDNN_SOFTMAX_ACCURATE;
            if (af == ACTIVE_FUNCTION_SOFTMAX_FAST) { softmax_flag = CUDNN_SOFTMAX_FAST; }
            CudaToolkit::setTensorDesc(cuda->tensor_desc_, 1, 1, X->row_, X->col_);
            auto s = cudnnSoftmaxBackward(cuda->cudnn_handle_, softmax_flag, CUDNN_SOFTMAX_MODE_INSTANCE,
                &const_real_1, cuda->tensor_desc_, A->data_, cuda->tensor_desc_, dA->data_, &const_real_0, cuda->tensor_desc_, dX->data_);
        }
        else
        {
            for (int i = 0; i < dX->col_; i++)
            {
                auto v = dotCol(A, i, dA, i);
                VectorMath::softmax_vb_sub(A->getDataPointer(0, i), dA->getDataPointer(0, i), v, dX->getDataPointer(0, i), dX->row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (X->cuda_type_ == CUDA_GPU)
        {
            CudaToolkit::setTensorDesc(cuda->tensor_desc_, 1, 1, X->row_, X->col_);
            auto s = cudnnSoftmaxBackward(cuda->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &const_real_1, cuda->tensor_desc_, A->data_, cuda->tensor_desc_, dA->data_, &const_real_0, cuda->tensor_desc_, dX->data_);
        }
        else
        {
            for (int i = 0; i < dX->col_; i++)
            {
                real v = 0;
                for (int j = 0; j < dX->row_; j++)
                {
                    v += dA->getData(i, j);
                }
                VectorMath::softmaxloss_vb_sub(A->getDataPointer(0, i), dA->getDataPointer(0, i), v, dX->getDataPointer(0, i), dX->row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_FINDMAX:
        //似乎应该是返回一个常数矩阵，若考虑效率应当留空此处在外部处理
        dX->initData(1);
        break;
    case ACTIVE_FUNCTION_SOFTPLUS:
        //该函数导数就是sigmoid
        activeForward(ACTIVE_FUNCTION_SIGMOID, X, dX);
        break;
    case ACTIVE_FUNCTION_SQUARE:
        Matrix::elementMul(dA, X, dX, 2);
        break;
    default:
        fprintf(stderr, "Parameters not enough!\n");
        break;
    }
}

//参数更多的的激活函数，包含了前面的功能，
//传附加参数的时候使用了C++11的初始化列表，因此效率可能较低，实际上如不考虑效率可以代替基本激活函数
//调用时请自己保证参数数量的正确性！
void MatrixExtend::activeForwardEx(ActiveFunctionType af, Matrix* X, Matrix* A,
    std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix*>& matrix_vector)
{
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = X->cuda_;
    switch (af)
    {
    default:
        activeForward(af, X, A);
        break;
    }
}

//参考activeForwardEx
void MatrixExtend::activeBackwardEx(ActiveFunctionType af, Matrix* A, Matrix* dA, Matrix* X, Matrix* dX,
    std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix*>& matrix_vector)
{
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = dA->cuda_;
    switch (af)
    {
    default:
        activeBackward(af, A, dA, X, dX);
        break;
    }
}


//池化，注意利用一个record记录下了对应位置
//gpu部分，平均模式下对padding的支持目前还有问题
void MatrixExtend::poolingForward(Matrix* X, Matrix* A, PoolingType pooling_type,
    int window_w, int window_h, int stride_w, int stride_h, int padding_w /*= 0*/, int padding_h /*= 0*/,
    real a /*= 1*/, real b /*= 0*/)
{
    auto cuda = X->cuda_;
    if (A->cuda_type_ == CUDA_GPU)
    {
        cudnnSetPooling2dDescriptor(cuda->pooling_desc_, cudnnPoolingMode_t(pooling_type), CUDNN_NOT_PROPAGATE_NAN,
            window_h, window_w, padding_h, padding_w, stride_h, stride_w);
        auto t = cudnnPoolingForward(cuda->cudnn_handle_, cuda->pooling_desc_, &a, X->tensor_desc_, X->data_, &b, A->tensor_desc_, A->data_);
        if (t)
        {
            fprintf(stderr, "POOL forward error %d\n", t);
        }
    }
    else
    {
        A->initData(0);
        for (int p = 0; p < A->number_ * A->channel_; p++)
        {
            for (int wA = 0; wA < A->width_; wA++)
            {
                for (int hA = 0; hA < A->height_; hA++)
                {
                    real v = 0;
                    if (pooling_type == POOLING_MAX) { v = -REAL_MAX; }
                    int n = 0;
                    int wX0 = wA * stride_w - padding_w;
                    int hX0 = hA * stride_h - padding_h;
                    for (int wX = wX0; wX < std::min(X->width_, wX0 + window_w); wX++)
                    {
                        for (int hX = hX0; hX < std::min(X->height_, hX0 + window_h); hX++)
                        {
                            if (pooling_type == POOLING_AVERAGE_PADDING || pooling_type == POOLING_AVERAGE_NOPADDING)
                            {
                                if (X->haveData(wX, hX, p, 0))
                                {
                                    v += X->getData(wX, hX, p, 0);
                                }
                                n++;
                            }
                            else if (pooling_type == POOLING_MAX)
                            {
                                if (X->haveData(wX, hX, p, 0))
                                {
                                    auto x = X->getData(wX, hX, p, 0);
                                    if (x > v)
                                    {
                                        v = x;
                                    }
                                }
                            }
                        }
                    }
                    if (pooling_type == POOLING_AVERAGE_PADDING)
                    {
                        v /= window_w * window_h;
                    }
                    else if (pooling_type == POOLING_AVERAGE_NOPADDING)
                    {
                        v /= n;
                    }
                    A->getData(wA, hA, p, 0) = v;
                }
            }
        }
    }
}

//使用cpu时利用了record -- 取消，直接计算
void MatrixExtend::poolingBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, PoolingType pooling_type,
    int window_w, int window_h, int stride_w, int stride_h, int padding_w /*= 0*/, int padding_h /*= 0*/,
    real a /*= 1*/, real b /*= 0*/)
{
    auto cuda = dA->cuda_;
    if (dX->cuda_type_ == CUDA_GPU)
    {
        //这个怎么看都快不了
        cudnnSetPooling2dDescriptor(cuda->pooling_desc_, cudnnPoolingMode_t(pooling_type), CUDNN_NOT_PROPAGATE_NAN,
            window_h, window_w, padding_h, padding_w, stride_h, stride_w);
        cudnnPoolingBackward(cuda->cudnn_handle_, cuda->pooling_desc_,
            &a, A->tensor_desc_, A->data_, dA->tensor_desc_, dA->data_, X->tensor_desc_, X->data_, &b, dX->tensor_desc_, dX->data_);
    }
    else
    {
        //dX->initData(0);
        for (int p = 0; p < dA->number_ * dA->channel_; p++)
        {
            for (int wdA = 0; wdA < dA->width_; wdA++)
            {
                for (int hdA = 0; hdA < dA->height_; hdA++)
                {
                    int wdX0 = wdA * stride_w - padding_w;
                    int hdX0 = hdA * stride_h - padding_h;
                    if (pooling_type == POOLING_MAX)
                    {
                        real max_v = -REAL_MAX;
                        real* max_p = nullptr;
                        for (int wdX = wdX0; wdX < std::min(dX->width_, wdX0 + window_w); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(dX->height_, hdX0 + window_h); hdX++)
                            {
                                if (X->haveData(wdX, hdX, p, 0))
                                {
                                    real v = X->getData(wdX, hdX, p, 0);
                                    if (v > max_v)
                                    {
                                        max_v = v;
                                        max_p = &dX->getData(wdX, hdX, p, 0);
                                    }
                                }
                            }
                        }
                        if (max_p)
                        {
                            *max_p = dA->getData(wdA, hdA, p, 0);
                        }
                    }
                    else
                    {
                        int n;
                        if (pooling_type == POOLING_AVERAGE_NOPADDING)
                        {
                            n = std::min(window_w, dX->width_ - wdA * stride_w) * std::min(window_h, dX->height_ - hdA * stride_h);
                        }
                        else
                        {
                            n = window_w * window_h;
                        }
                        real v = dA->getData(wdA, hdA, p, 0) / n;
                        for (int wdX = wdX0; wdX < std::min(dX->width_, wdX0 + window_w); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(dX->height_, hdX0 + window_h); hdX++)
                            {
                                if (dX->haveData(wdX, hdX, p, 0))
                                {
                                    dX->getData(wdX, hdX, p, 0) += a * v + b * dX->getData(wdX, hdX, p, 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//卷积就是有连接就算
//这个循环顺序次数最少
#define CONV_OPERATION1(X, W, A, wX, hX, wW, hW, wA, hA, DO_SOMETHING) \
    do {\
        for (int wA = 0; wA < A->width_; wA++)\
            for (int hA = 0; hA < A->height_; hA++)\
                for (int wW = 0; wW < W->width_; wW++)\
                    for (int hW = 0; hW < W->height_; hW++)\
                    {\
                        int wX = wA + wW;\
                        int hX = hA + hW;\
                        { DO_SOMETHING }\
                    }\
    } while(0)

//前向卷积
//辅助数组若空间不够会调整大小，如果辅助数组空间足够，则认为已经初始化过
//从外部引入辅助空间目的是降低初始化的次数
//当使用CUDA计算时，不需要辅助转换的整数数组，这时该数组会被初始化为两个元素，分别为算法和所需的工作空间大小，在首次计算的时候完成
void MatrixExtend::convolutionForward(Matrix* X, Matrix* A, Matrix* W, Matrix* workspace, std::vector<int>& workspace_forward,
    int stride_w, int stride_h, int padding_w, int padding_h, real a, real b)
{
    auto cuda = X->cuda_;
    if (A->cuda_type_ == CUDA_GPU)
    {
        //cudnnSetConvolutionMathType(cuda->convolution_desc_, CUDNN_TENSOR_OP_MATH);    //从volta开始支持
        auto scd = cudnnSetConvolution2dDescriptor(cuda->convolution_desc_, padding_h, padding_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, MYCUDNN_DATA_REAL);
        auto sfd = cudnnSetFilter4dDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, A->channel_, X->channel_, W->height_, W->width_);

        //寻找最快的算法
        if (workspace_forward.empty())
        {
            int n;
            workspace_forward.resize(1);
            cudnnConvolutionFwdAlgoPerf_t cfap[8];
            auto t = cudnnFindConvolutionForwardAlgorithm(cuda->cudnn_handle_, X->tensor_desc_,
                cuda->filter_desc_, cuda->convolution_desc_, A->tensor_desc_, 8, &n, cfap);
            workspace_forward[0] = cfap[0].algo;
            if (cfap[0].memory > workspace->getDataSizeInByte())
            {
                workspace->resize(1, 1, 1, cfap[0].memory / 4 + 1);
            }
            if (t)
            {
                fprintf(stderr, "%d, %d, %d\n", scd, sfd, t);
            }
        }
        auto cfa = cudnnConvolutionFwdAlgo_t(workspace_forward[0]);
        auto scf = cudnnConvolutionForward(cuda->cudnn_handle_, &a, X->tensor_desc_, X->data_, cuda->filter_desc_, W->data_,
            cuda->convolution_desc_, cfa, workspace->data_, workspace->getDataSizeInByte(), &b, A->tensor_desc_, A->data_);
        if (scf)
        {
            fprintf(stderr, "CONV forward error %d, %d, %d\n", scd, sfd, scf);
        }
    }
    else
    {
        A->initData(0);
        //辅助矩阵的尺寸
        int row = A->width_ * A->height_;
        int col = X->channel_ * W->width_ * W->height_;
        auto X_ex = new Matrix(row, col);
        if (workspace_forward.size() < row * col)
        {
            workspace_forward.resize(row * col);
            //ex_pos记录展开的位置，预先记录节省时间
            for (int cX = 0; cX < X->channel_; cX++)
            {
                CONV_OPERATION1(X, W, A, wX, hX, wW, hW, wA, hA,
                {
                    //记录X_ex中每个位置对应X中的元素，为了效率将二者都拍扁
                    int pA = A->whcn2i(wA, hA, 0, 0);    //X_ex中对应的行，即在A中的位置
                    int pW = W->whcn2i(wW, hW, cX, 0);   //X_ex中对应的列，即在W中的位置
                    //拍扁
                    int pX = X->whcn2i(wX, hX, cX, 0);   //X其中一组特征对应的位置
                    int pX_ex = X_ex->mn2i(pA, pW);
                    workspace_forward[pX_ex] = pX;             //记录展开的位置
                });
            }
        }
        auto X_sub = new Matrix(X->row_, 1, MATRIX_DATA_OUTSIDE);
        auto A_sub = new Matrix(A->width_ * A->height_, A->channel_, MATRIX_DATA_OUTSIDE);
        for (int i = 0; i < X->number_; i++)
        {
            X_sub->shareData(X, 0, i);
            A_sub->shareData(A, 0, i);
            for (int j = 0; j < X_ex->getDataSize(); j++)
            {
                X_ex->getData(j) = X_sub->getData(workspace_forward[j]);
            }
            MatrixExtend::mul(X_ex, W, A_sub, a, b);
        }
        delete X_sub;
        delete A_sub;
        delete X_ex;

#ifdef DIRECT_COMPUTE_CONVOLUTION
        //fprintf(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        //直接计算前向卷积，仅用于CPU，速度比较慢，应废弃
        for (int n = 0; n < A->number_; n++)
        {
            for (int cX = 0; cX < X->channel_; cX++)
            {
                for (int cA = 0; cA < A->channel_; cA++)
                {
                    CONV_OPERATION1(X, W, A, wX, hX, wW, hW, wA, hA,
                    {
                        A->getData(wA, hA, cA, n) += X->getData(wX, hX, cX, n) * W->getData(wW, hW, cX, cA);
                    });
                }
            }
        }
#endif
    }
}

//计算dX只需要W，dA；计算dW只需要X，dA；计算dB只需要dA
void MatrixExtend::convolutionBackward(Matrix* A, Matrix* dA, Matrix* X, Matrix* dX, Matrix* W, Matrix* dW, Matrix* dB,
    Matrix* workspace, std::vector<int>& workspace_backward_dx, std::vector<int>& workspace_backward_dw,
    int stride_w, int stride_h, int padding_w, int padding_h, real a, real b)
{
    auto cuda = dA->cuda_;
    //这里不用dX判断是因为dX可能是空
    if (dA->cuda_type_ == CUDA_GPU)
    {
        cudnnSetConvolution2dDescriptor(cuda->convolution_desc_, padding_h, padding_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, MYCUDNN_DATA_REAL);
        auto _A = A, _X = X, _W = W;
        if (A == nullptr) { _A = dA; }
        if (X == nullptr) { _X = dX; }
        if (W == nullptr) { _W = dW; }
        if (dX)
        {
            cudnnSetFilter4dDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, _A->channel_, _X->channel_, _W->height_, _W->width_);
            //寻找最快的算法
            if (workspace_backward_dx.empty())
            {
                int n;
                workspace_backward_dx.resize(1);
                cudnnConvolutionBwdDataAlgoPerf_t cbdap[8];
                auto t = cudnnFindConvolutionBackwardDataAlgorithm(cuda->cudnn_handle_, cuda->filter_desc_,
                    dA->tensor_desc_, cuda->convolution_desc_, dX->tensor_desc_, 8, &n, cbdap);
                workspace_backward_dx[0] = cbdap[0].algo;
                if (cbdap[0].memory > workspace->getDataSizeInByte())
                {
                    workspace->resize(1, 1, 1, cbdap[0].memory / 4 + 1);
                }
            }
            auto cbda = cudnnConvolutionBwdDataAlgo_t(workspace_backward_dx[0]);
            auto scbx = cudnnConvolutionBackwardData(cuda->cudnn_handle_, &a, cuda->filter_desc_, W->data_, dA->tensor_desc_, dA->data_,
                cuda->convolution_desc_, cbda, workspace->data_, workspace->getDataSizeInByte(), &b, dX->tensor_desc_, dX->data_);
        }
        if (dW)
        {
            cudnnSetFilter4dDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, _A->channel_, _X->channel_, _W->height_, _W->width_);
            //寻找最快的算法
            if (workspace_backward_dw.empty())
            {
                int n;
                workspace_backward_dw.resize(1);
                cudnnConvolutionBwdFilterAlgoPerf_t cbfap[8];
                cudnnFindConvolutionBackwardFilterAlgorithm(cuda->cudnn_handle_, X->tensor_desc_, dA->tensor_desc_,
                    cuda->convolution_desc_, cuda->filter_desc_, 8, &n, cbfap);
                workspace_backward_dw[0] = cbfap[0].algo;
                if (cbfap[0].memory > workspace->getDataSizeInByte())
                {
                    workspace->resize(1, 1, 1, cbfap[0].memory / 4 + 1);
                }
            }
            auto cbfa = cudnnConvolutionBwdFilterAlgo_t(workspace_backward_dw[0]);

            auto scbw = cudnnConvolutionBackwardFilter(cuda->cudnn_handle_, &a, X->tensor_desc_, X->data_, dA->tensor_desc_, dA->data_,
                cuda->convolution_desc_, cbfa, workspace->data_, workspace->getDataSizeInByte(), &b, cuda->filter_desc_, dW->data_);
            if (scbw)
            {
                fprintf(stderr, "CONV backward error %d\n", scbw);
            }
        }
        if (dB)
        {
            auto scbb = cudnnConvolutionBackwardBias(cuda->cudnn_handle_, &a, dA->tensor_desc_, dA->data_, &b, dB->tensor_desc_, dB->data_);
        }
    }
    else
    {
        if (dX)
        {
            //计算dX从数学上来看可以反向来求展开后的dX，再压缩，但是看起来加法次数较多
            //转置W的输入和输出
            auto W2 = new Matrix(W->width_, W->height_, W->number_, W->channel_);
            for (int i = 0; i < W->channel_; i++)
            {
                for (int j = 0; j < W->number_; j++)
                {
                    copyDataPointer(W, W->getDataPointer(0, 0, i, j), W2, W2->getDataPointer(0, 0, j, i), W->width_ * W->height_);
                }
            }
            //辅助矩阵的尺寸
            int row = dX->width_ * dX->height_;
            int col = dA->channel_ * W->width_ * W->height_;
            auto dA_ex = new Matrix(row, col);
            dA_ex->initData(0);
            if (workspace_backward_dx.size() < row * col)
            {
                workspace_backward_dx.resize(row * col);
                for (int i = 0; i < workspace_backward_dx.size(); i++) { workspace_backward_dx[i] = -1; }
                for (int cA = 0; cA < dA->channel_; cA++)
                {
                    CONV_OPERATION1(dX, W, dA, wX, hX, wW, hW, wA, hA,
                    {
                        int pX = dX->whcn2i(wX, hX, 0, 0);
                        //这里用W或者W2没有区别
                        int pW = W2->whcn2i(wW, hW, cA, 0);
                        //拍扁
                        int pA = dA->whcn2i(wA, hA, cA, 0);
                        int p_ex = dA_ex->mn2i(pX, pW);
                        workspace_backward_dx[p_ex] = pA;
                    });
                }
            }
            auto dA_sub = new Matrix(dA->row_, 1, MATRIX_DATA_OUTSIDE);
            auto dX_sub = new Matrix(dX->width_ * dX->height_, dX->channel_, MATRIX_DATA_OUTSIDE);
            for (int i = 0; i < dA->number_; i++)
            {
                dA_sub->shareData(dA, 0, i);
                dX_sub->shareData(dX, 0, i);
                for (int j = 0; j < dA_ex->getDataSize(); j++)
                {
                    if (workspace_backward_dx[j] >= 0)
                    {
                        dA_ex->getData(j) = dA_sub->getData(workspace_backward_dx[j]);
                    }
                }
                MatrixExtend::mul(dA_ex, W2, dX_sub, a, b);
            }
            delete W2;
            delete dX_sub;
            delete dA_sub;
            delete dA_ex;
        }
        //暂时如此写，看情况能否跟上面合并
        if (dW)
        {
            dW->scale(b);
            //辅助矩阵的尺寸
            int row = dW->width_ * dW->height_ * dW->channel_;
            int col = dA->width_ * dA->height_;
            auto X_ex = new Matrix(row, col);
            X_ex->initData(0);
            if (workspace_backward_dw.size() < row * col)
            {
                workspace_backward_dw.resize(row * col);
                for (int i = 0; i < workspace_backward_dw.size(); i++) { workspace_backward_dw[i] = -1; }
                //cW==cX, nW=cA
                for (int cW = 0; cW < dW->channel_; cW++)
                {
                    CONV_OPERATION1(X, dW, dA, wX, hX, wW, hW, wA, hA,
                    {
                        int pW = dW->whcn2i(wW, hW, cW, 0);
                        int pA = dA->whcn2i(wA, hA, 0, 0);
                        //拍扁
                        int pX = X->whcn2i(wX, hX, cW, 0);
                        int p_ex = X_ex->mn2i(pW, pA);
                        workspace_backward_dw[p_ex] = pX;
                    });
                }
            }
            auto dA_sub = new Matrix(dA->width_ * dA->height_, dA->channel_, MATRIX_DATA_OUTSIDE);
            auto X_sub = new Matrix(X->row_, 1, MATRIX_DATA_OUTSIDE);
            for (int i = 0; i < dA->number_; i++)
            {
                dA_sub->shareData(dA, 0, i);
                X_sub->shareData(X, 0, i);
                for (int j = 0; j < X_ex->getDataSize(); j++)
                {
                    //if ((*ex2)[j] >= 0) //因为是满的不需要
                    X_ex->getData(j) = X_sub->getData(workspace_backward_dw[j]);
                }
                MatrixExtend::mul(X_ex, dA_sub, dW, a, b);
            }
            delete X_sub;
            delete dA_sub;
            delete X_ex;
        }
        if (dB)
        {
            dB->scale(b);
            //这个就是对对应的dA求和
            for (int n = 0; n < dA->number_; n++)
            {
                for (int c = 0; c < dA->channel_; c++)
                {
                    dB->getData(0, 0, c, 0) += a * VectorMath::sum(dA->getDataPointer(0, 0, c, n), dA->width_ * dA->height_);
                }
            }
        }

#ifdef DIRECT_COMPUTE_CONVOLUTION
        //这一段是为了参考保留，不要打开这段代码
        //fprintf(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        if (dW) { dW->scale(b); }
        if (dX) { dX->scale(b); }
        //直接计算反向卷积，速度较慢
        for (int n = 0; n < A->number_; n++)
        {
            for (int cX = 0; cX < X->channel_; cX++)
            {
                for (int cA = 0; cA < A->channel_; cA++)
                {
                    CONV_OPERATION1(X, W, A, wX, hX, wW, hW, wA, hA,
                    {
                        if (dX)
                        {
                            dX->getData(wX, hX, cX, n) += a * dA->getData(wA, hA, cA, n) * W->getData(wW, hW, cX, cA);
                        }
                        if (dW)
                        {
                            dW->getData(wW, hW, cX, cA) += a * X->getData(wX, hX, cX, n) * dA->getData(wA, hA, cA, n);
                        }
                    });
                }
            }
        }
#endif
    }
}

void MatrixExtend::adaUpdate(Matrix* E_dw2, Matrix* E_g2, Matrix* dw, real rou, real epsilon)
{
    if (dw->cuda_type_ == CUDA_GPU)
    {
        cuda_ada_update(E_dw2->data_, E_g2->data_, dw->data_, dw->data_size_, rou, epsilon);
    }
    else
    {
        auto& p1 = E_dw2->data_;
        auto& p2 = E_g2->data_;
        auto& p3 = dw->data_;
        for (int i = 0; i < dw->data_size_; i++)
        {
            p2[i] = p2[i] * rou + p3[i] * p3[i] * (1 - rou);
            p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
            p1[i] = p1[i] * rou + p3[i] * p3[i] * (1 - rou);
        }
    }
}

void MatrixExtend::adaDeltaUpdate(Matrix* E_g2, Matrix* E_d2, Matrix* g, Matrix* d, real rou, real epsilon)
{
    if (g->cuda_type_ == CUDA_GPU)
    {
        cuda_adaDelta_update(E_g2->data_, E_d2->data_, g->data_, d->data_, g->data_size_, rou, epsilon);
    }
    else
    {
        auto& p1 = E_g2->data_;
        auto& p2 = E_d2->data_;
        auto& p3 = g->data_;
        auto& p4 = d->data_;
        for (int i = 0; i < g->data_size_; i++)
        {
            p1[i] = p1[i] * rou + p3[i] * p3[i] * (1 - rou);
            p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
            p2[i] = p2[i] * rou + p4[i] * p4[i] * (1 - rou);
        }
    }
}

