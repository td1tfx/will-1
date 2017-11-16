#include "Matrix.h"
#include "VectorMath.h"

const real Matrix::const_real_1 = 1;
const real Matrix::const_real_0 = 0;

//普通二维矩阵构造函数
Matrix::Matrix(int m, int n, MatrixDataType try_inside, CudaType try_cuda)
{
    matrix_data_ = try_inside;
    cuda_type_ = (try_cuda == CUDA_GPU) && (CudaToolkit::getCudaState() == CUDA_GPU) ? CUDA_GPU : CUDA_CPU;
    if (cuda_type_ == CUDA_GPU)
    {
        cuda_ = CudaToolkit::select(CudaToolkit::getCurrentDevice());
    }
    row_ = m;
    col_ = n;
    width_ = 1;
    height_ = 1;
    channel_ = m;
    number_ = n;
    data_size_ = int64_t(row_) * int64_t(col_);
    if (matrix_data_ == MATRIX_DATA_INSIDE && data_size_ > 0)
    {
        data_ = mallocData(data_size_);
        //initData(0);
        occupy_data_size_ = data_size_;
    }
    if (cuda_type_ == CUDA_GPU)
    {
        CudaToolkit::cudnnCreateDescriptor(&tensor_desc_);
        CudaToolkit::setTensorDesc(tensor_desc_, width_, height_, channel_, number_);
    }
    //nvml_id_ = cuda_->nvml_id_;
}

//4阶张量形式构造函数，用于池化和卷积
//当矩阵是张量时，实际上原本的列数毫无意义，但是无论张量还是矩阵，组数都是最后一维
Matrix::Matrix(int w, int h, int c, int n, MatrixDataType try_inside /*= MatrixData_Inside*/, CudaType try_cuda /*= Cuda_GPU*/)
    : Matrix(w * h * c, n, try_inside, try_cuda)
{
    width_ = w;
    height_ = h;
    channel_ = c;
    number_ = n;
    if (cuda_->global_cuda_type_ == CUDA_GPU)
    {
        CudaToolkit::setTensorDesc(tensor_desc_, w, h, c, n);
    }
}

//根据已知矩阵的维度创建一个新矩阵
Matrix::Matrix(Matrix* src, MatrixDataType try_inside, CudaType try_cuda)
    : Matrix(src->width_, src->height_, src->channel_, src->number_, try_inside, try_cuda)
{

}

Matrix::Matrix(int w, int h, int c, int n, real* data, CudaType try_cuda /*= CUDA_GPU*/)
    : Matrix(w * h * c, n, MATRIX_DATA_OUTSIDE, try_cuda)
{
    data_ = data;
}

Matrix* Matrix::clone(MatrixDataType try_inside /*= MatrixData_Inside*/, CudaType try_cuda /*= Cuda_GPU*/)
{
    auto M = new Matrix(this, try_inside, try_cuda);
    copyData(this, M);
    return M;
}

Matrix* Matrix::cloneShared()
{
    auto M = new Matrix(this, MATRIX_DATA_OUTSIDE, cuda_type_);
    M->shareData(this, 0, 0);
    return M;
}

Matrix* Matrix::cloneSharedCol(int col)
{
    auto M = new Matrix(width_, height_, channel_, col, MATRIX_DATA_OUTSIDE, cuda_type_);
    M->shareData(this, 0, 0);
    return M;
}

Matrix::~Matrix()
{
    if (matrix_data_ == MATRIX_DATA_INSIDE) { freeData(); }
    CudaToolkit::cudnnDestroyDescriptor(tensor_desc_);
}

//返回值：-1空矩阵，未重新分配内存，1重新分配内存
//若只有一个数值参数，则仅重新处理组数
int Matrix::resize(int n, bool force /*= false*/)
{
    if (this == nullptr) { return -1; }
    return resize(width_, height_, channel_, n, force);
}

int Matrix::resize(int m, int n, bool force /*= false*/)
{
    return resize(1, 1, m, n, force);
}

int Matrix::resize(int w, int h, int c, int n, bool force /*= false*/)
{
    if (this == nullptr) { return -1; }
    row_ = w * h * c;
    col_ = n;
    data_size_ = row_ * col_;
    width_ = w;
    height_ = h;
    channel_ = c;
    number_ = n;
    CudaToolkit::setTensorDesc(tensor_desc_, width_, height_, channel_, number_);
    //空间不够或者强制则重新分配
    if (data_size_ > occupy_data_size_ || force)
    {
        //重新申请空间
        if (matrix_data_ == MATRIX_DATA_INSIDE)
        {
            auto temp = mallocData(data_size_);
            if (data_)
            {
                copyDataPointer(this, data_, this, temp, std::min(data_size_, occupy_data_size_));
            }
            freeData();
            data_ = temp;
            occupy_data_size_ = data_size_;
        }
        return 1;
    }
    return 0;
}

int Matrix::resize(Matrix* X, bool force /*= false*/)
{
    return resize(X->width_, X->height_, X->channel_, X->number_, force);
}

//输出矩阵内容
//注意这个实际上是按照内存顺序
void Matrix::print(FILE* fout)
{
    if (this == nullptr) { return; }
    auto temp = mallocCPU_dataToCPU();
    for (int p = 0; p < channel_ * number_; p++)
    {
        for (int h = 0; h < height_; h++)
        {
            for (int w = 0; w < width_; w++)
            {
                auto v = temp[whcn2i(w, h, p, 0)];
                fprintf(fout, "%4.2g ", v);
            }
            fprintf(fout, "\n");
        }
        fprintf(fout, "\n");
    }
    freeCPU(temp);
}

int Matrix::save(FILE* fout /*= stdout*/)
{
    auto temp = mallocCPU_dataToCPU();
    fwrite(temp, sizeof(real), data_size_, fout);
    freeCPU(temp);
    return data_size_;
}

//从一组实数指针载入矩阵内容
//int Matrix::load(deque2<real>& v)
//{
//    if (this == nullptr) { return 0; }
//    auto temp = mallocCPU();
//    int k = 0;
//    for (int p = 0; p < channel_ * number_; p++)
//    {
//        for (int h = 0; h < height_; h++)
//        {
//            for (int w = 0; w < width_; w++)
//            {
//                if (v.empty()) { break; }
//                temp[whcn2i(w, h, p, 0)] = v.get_pop_front();
//                k++;
//            }
//        }
//    }
//    dataToGPU_freeCPU(temp);
//    return k;
//}

int Matrix::load(FILE* fin /*= stdout*/)
{
    auto temp = mallocCPU();
    fread(temp, sizeof(real), data_size_, fin);
    dataToGPU_freeCPU(temp);
    return data_size_;
}

//将矩阵当做向量，按照内存中的顺序依次输出
void Matrix::printAsVector(FILE* fout /*= stdout*/)
{
    auto temp = mallocCPU_dataToCPU();
    for (int i = 0; i < data_size_; i++)
    {
        fprintf(fout, "%14.11g ", temp[i]);
    }
    fprintf(fout, "\n");
    freeCPU(temp);
}

//将矩阵当做向量，按照内存中的顺序依次载入
//int Matrix::loadAsVector(std::deque2<real>& v)
//{
//    auto temp = mallocCPU();
//    int k = 0;
//    for (int i = 0; i < row_; i++)
//    {
//        if (v.empty()) { break; }
//        temp[i] = v.get_pop_front();
//    }
//    dataToGPU_freeCPU(temp);
//    return k;
//}

//按照矩阵输出，因为是列优先，故不是内存顺序
void Matrix::printAsMatrix(FILE* fout /*= stdout*/)
{
    auto temp = mallocCPU_dataToCPU();
    for (int r = 0; r < row_; r++)
    {
        for (int c = 0; c < col_; c++)
        {
            auto v = temp[mn2i(r, c)];
            fprintf(fout, "%g ", v);
        }
        fprintf(fout, "\n");
    }
    fprintf(fout, "\n");
    freeCPU(temp);
}

//将外界的值复制到矩阵，参数指针必须指向Host内存！
void Matrix::copyDataInFromHost(real* src, int64_t size)
{
    if (cuda_type_ == CUDA_GPU)
    {
        cudaMemcpy(data_, src, int(sizeof(real)*std::min(size, data_size_)), cudaMemcpyHostToDevice);
    }
    else
    {
        memcpy(data_, src, int(sizeof(real)*std::min(size, data_size_)));
    }
}

//将矩阵的值复制到外界，参数指针必须指向Host内存！
void Matrix::copyDataOutToHost(real* dst, int64_t size)
{
    if (cuda_type_ == CUDA_GPU)
    {
        cudaMemcpy(dst, data_, int(sizeof(real)*std::min(size, data_size_)), cudaMemcpyDeviceToHost);
    }
    else
    {
        memcpy(dst, data_, int(sizeof(real)*std::min(size, data_size_)));
    }
}

//复制数据，只处理较少的
void Matrix::copyData(Matrix* A, Matrix* R, int64_t size /*= -1*/)
{
    copyDataPointer(A, A->data_, R, R->data_, size);
}

//警告：乱用模式会受惩罚！
void Matrix::copyDataPointer(Matrix* A, real* A_pointer, Matrix* R, real* R_pointer, int64_t size /*= -1*/)
{
    if (A_pointer == R_pointer) { return; }

    auto a_type = A == nullptr ? CUDA_CPU : A->cuda_type_;
    auto r_type = R == nullptr ? CUDA_CPU : R->cuda_type_;

    if (size < 0)
    {
        size = std::min(A->getDataSize(), R->getDataSize());
    }
    size *= sizeof(real);

    int state = cudaSuccess;
    if (r_type == CUDA_GPU && a_type == CUDA_GPU)
    {
        state = cudaMemcpy(R_pointer, A_pointer, size, cudaMemcpyDeviceToDevice);
    }
    else if (r_type == CUDA_GPU && a_type == CUDA_CPU)
    {
        state = cudaMemcpy(R_pointer, A_pointer, size, cudaMemcpyHostToDevice);
    }
    else if (r_type == CUDA_CPU && a_type == CUDA_GPU)
    {
        state = cudaMemcpy(R_pointer, A_pointer, size, cudaMemcpyDeviceToHost);
    }
    else
    {
        memcpy(R_pointer, A_pointer, size);
    }
    if (state != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed with error code is %d, size is %lld!!!\n", state, size);
    }
}

void Matrix::copyDataAcrossDevice(Matrix* A, Matrix* R, int64_t size /*= -1*/)
{
    if (size < 0)
    {
        size = std::min(A->getDataSize(), R->getDataSize());
    }
    size *= sizeof(real);
    if (R->cuda_type_ == CUDA_GPU && A->cuda_type_ == CUDA_GPU)
    {
        int state = cudaMemcpyPeer(R->data_, CudaToolkit::getCudaDeviceFromNvml(R->cuda_->nvml_id_), A->data_, CudaToolkit::getCudaDeviceFromNvml(A->cuda_->nvml_id_), size);
        if (state != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpyPeer failed with error code is %d, size is %lld!!!\n", state, size);
        }
    }
    else
    {
        copyDataPointer(A, A->data_, R, R->data_, size);
    }
}

//尝试将内存矩阵变为显存矩阵
void Matrix::toGPU()
{
    if (cuda_->global_cuda_type_ == CUDA_GPU)
    {
        if (cuda_type_ == CUDA_CPU)
        {
            cuda_type_ = CUDA_GPU;
            auto temp = mallocData(occupy_data_size_);
            if (temp)
            {
                std::swap(temp, data_);
                dataToGPU_freeCPU(temp);
            }
            else
            {
                cuda_type_ = CUDA_CPU;
            }
        }
    }
}

//need_data: 将显存中的数据转移到内存
//丢弃显存中的数据，在内存中重新分配同等大小空间
void Matrix::toCPU(bool reserve_data /*= true*/)
{
    if (cuda_type_ == CUDA_GPU)
    {
        real* temp;
        if (reserve_data)
        {
            temp = mallocCPU_dataToCPU();
        }
        else
        {
            temp = mallocCPU();
        }
        if (temp)
        {
            std::swap(temp, data_);
            cudaFree(temp);
        }
        cuda_type_ = CUDA_CPU;
    }
}

//将一个外部数据矩阵的指针指向其他位置
void Matrix::shareData(Matrix* A, int m, int n)
{
    if (cuda_ != A->cuda_)
    {
        fprintf(stderr, "Error: share data are in different device!!!\n");
    }
    if (matrix_data_ == MATRIX_DATA_OUTSIDE && cuda_type_ == A->cuda_type_)
    {
        data_ = A->getDataPointer(m, n);
    }
}

void Matrix::shareData(Matrix* A, int w, int h, int c, int n)
{
    if (matrix_data_ == MATRIX_DATA_OUTSIDE && cuda_type_ == A->cuda_type_)
    {
        data_ = A->getDataPointer(w, h, c, n);
    }
}

void Matrix::shareData(real* data)
{
    if (matrix_data_ == MATRIX_DATA_OUTSIDE)
    {
        data_ = data;
    }
}

//flip和transpose暂时仅用于cpu
void Matrix::filp(int flip_flag)
{
    if (cuda_type_ == CUDA_GPU) { return; }
    auto temp = new Matrix(width_, height_, MATRIX_DATA_INSIDE, CUDA_CPU);
    for (int c = 0; c < channel_; c++)
    {
        for (int n = 0; n < number_; n++)
        {
            Matrix::copyDataPointer(this, getDataPointer(0, 0, c, n), temp, temp->getDataPointer());
            switch (flip_flag)
            {
            case 1:
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        getData(i, j, c, n) = temp->getData(width_ - 1 - i, j);
                    }
                }
                break;
            case 0:
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        getData(i, j, c, n) = temp->getData(i, height_ - 1 - j);
                    }
                }
                break;
            case -1:
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        getData(i, j, c, n) = temp->getData(width_ - 1 - i, height_ - 1 - j);
                    }
                }
                break;
            default:
                break;
            }
        }
    }
    delete temp;
}

void Matrix::transpose(int transpose_flag)
{
    if (cuda_type_ == CUDA_GPU) { return; }
    if (transpose_flag == 0 || width_ != height_) { return; }
    auto temp = new Matrix(width_, height_, MATRIX_DATA_INSIDE, CUDA_CPU);
    for (int c = 0; c < channel_; c++)
    {
        for (int n = 0; n < number_; n++)
        {
            Matrix::copyDataPointer(this, getDataPointer(0, 0, c, n), temp, temp->getDataPointer());
            for (int i = 0; i < width_; i++)
            {
                for (int j = 0; j < height_; j++)
                {
                    getData(i, j, c, n) = temp->getData(j, i);
                }
            }
        }
    }
    delete temp;
}

//根据矩阵的cuda属性分配内存或显存，会保留数据
real* Matrix::mallocData(int64_t size)
{
    if (cuda_type_ == CUDA_GPU)
    {
        real* d = nullptr;
        if (cudaMalloc((void**)&d, size * sizeof(real)) == cudaSuccess)
        {
            //dataIsWhere = DataInDevice;
        }
        else
        {
            fprintf(stderr, "Matrix malloc data failed! size is %g\n", 1.0 * size * sizeof(real));
        }
        return d;
    }
    else
    {
        return new real[size];
    }
}

//销毁矩阵的数据指针
void Matrix::freeData()
{
    if (data_ == nullptr) { return; }
    if (cuda_type_ == CUDA_GPU)
    {
        cudaFree(data_);
        return;
    }
    else
    {
        delete data_;
    }
    data_ = nullptr;
}

//生成一个指针用于操作cuda设备中的数据，并将数据读下来
real* Matrix::mallocCPU_dataToCPU()
{
    if (cuda_type_ == CUDA_GPU)
    {
        auto temp = new real[data_size_];
        cudaMemcpy(temp, data_, sizeof(real)*data_size_, cudaMemcpyDeviceToHost);
        return temp;
    }
    else
    {
        return data_;
    }
}

//销毁从cuda读取到内存的数据指针
void Matrix::freeCPU(real* temp)
{
    if (cuda_type_ == CUDA_GPU)
    {
        delete temp;
    }
}

//生成一个指针用于操作cuda设备中的数据，仅准备不读取
real* Matrix::mallocCPU()
{
    if (cuda_type_ == CUDA_GPU)
    {
        return new real[data_size_];
    }
    else
    {
        return data_;
    }
}

//将内存中的数据传到cuda设备并销毁内存指针
void Matrix::dataToGPU_freeCPU(real* temp)
{
    if (cuda_type_ == CUDA_GPU)
    {
        cudaMemcpy(data_, temp, sizeof(real) * data_size_, cudaMemcpyHostToDevice);
        delete temp;
    }
}

//将前面几列复制到整个矩阵
void Matrix::repeat(int c/*=1*/)
{
    if (cuda_type_ == CUDA_GPU)
    {
        for (int i = c; i < col_; i *= 2)
        {
            cudaMemcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(real) * row_ * std::min(i, col_ - i), cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        //#pragma loop(hint_parallel(8))
        for (int i = c; i < col_; i *= 2)
        {
            memcpy(getDataPointer(0, i), getDataPointer(0, 0), sizeof(real) * row_ * std::min(i, col_ - i));
        }
    }
}

//一列中最大值的序号
int Matrix::indexColMaxAbs(int c)
{
    if (cuda_type_ == CUDA_GPU)
    {
        return cuda_->cublas_->iamax(row_, getDataPointer(0, c), 1);
    }
    else
    {
        return Cblas::iamax(row_, getDataPointer(0, c), 1);
    }
}

//绝对值求和（直接调用的blas，注意这里实际上需要的功能只是求和）
real Matrix::sumAbs()
{
    if (cuda_type_ == CUDA_GPU)
    {
        return cuda_->cublas_->asum(data_size_, data_, 1);
    }
    else
    {
        return Cblas::asum(data_size_, data_, 1);
    }
}

//一列的绝对值和
real Matrix::sumColAbs(int c)
{
    if (cuda_type_ == CUDA_GPU)
    {
        return cuda_->cublas_->asum(row_, getDataPointer(0, c), 1);
    }
    else
    {
        return Cblas::asum(row_, getDataPointer(0, c), 1);
    }
}

real Matrix::sum()
{
    auto temp1 = new Matrix(data_size_, 1);
    temp1->initData(1);
    real r = dot(this, temp1);
    delete temp1;
    return r;
}

//以同一个值初始化矩阵
//inc不为零时仅用于测试，不要用于实际计算！
void Matrix::initData(real v, int inc/*=0*/)
{
    if (!data_) { return; }
    if (cuda_type_ == CUDA_GPU && inc == 0)
    {
        if (cuda_type_ == CUDA_GPU)
        {
            cudnnSetTensor(cuda_->cudnn_handle_, tensor_desc_, data_, &v);
        }
    }
    else
    {
        auto temp = mallocCPU();
        //#pragma loop(hint_parallel(8))
        if (v == 0 && inc == 0)
        {
            memset(data_, 0, getDataSizeInByte());
        }
        else
        {
            for (int i = 0; i < data_size_; i++)
            {
                temp[i] = i * inc + v;
            }
        }
        dataToGPU_freeCPU(temp);
    }
}


//随机数初始化矩阵，注意这个函数调用次数很少
void Matrix::initRandom(int seed /*= 0*/)
{
    if (!data_) { return; }
    Random<real> r;
    r.set_seed(seed);
    initRandom(&r);
}

void Matrix::initRandom(Random<real>* r)
{
    if (!data_) { return; }
    auto temp = mallocCPU();
    for (int i = 0; i < data_size_; i++)
    {
        temp[i] = r->rand();
        //fprintf(stderr, "%f", temp[i]);
    }
    dataToGPU_freeCPU(temp);
}

void Matrix::sectionLimit(real v0, real v1)
{
    if (cuda_type_ == CUDA_GPU)
    {
        cuda_sectionlimit(data_, nullptr, data_, data_size_, v0, v1);
    }
    else
    {
        for (int i = 0; i < data_size_; i++)
        {
            data_[i] = std::min(data_[i], v1);
            data_[i] = std::max(data_[i], v0);
        }
    }
}

//数乘
void Matrix::scale(real v)
{
    if (v == 1) { return; }
    if (v == 0)
    {
        initData(0);
        return;
    }
    if (cuda_type_ == CUDA_GPU)
    {
        cuda_->cublas_->scal(data_size_, v, data_, 1);
    }
    else
    {
        Cblas::scal(data_size_, v, data_, 1);
    }
}

//选择一列数乘
void Matrix::scaleCol(real v, int c)
{
    if (v == 1) { return; }
    if (cuda_type_ == CUDA_GPU)
    {
        cuda_->cublas_->scal(row_, v, getDataPointer(0, c), 1);
    }
    else
    {
        Cblas::scal(row_, v, getDataPointer(0, c), 1);
    }
}

//矩阵乘，R = aAB+cR
void Matrix::mul(Matrix* A, Matrix* B, Matrix* R, real a /*= 1*/, real c /*= 0*/, MatrixTransType ta /*= NoTrans*/, MatrixTransType tb /*= NoTrans*/)
{
    int m = R->row_;
    int n = R->col_;
    int lda = A->row_;
    int k = A->col_;
    int ldb = B->row_;
    if (ta == MATRIX_TRANS) { k = A->row_; }
    if (R->cuda_type_ == CUDA_GPU)
    {
        R->cuda_->cublas_->gemm(ta, tb, m, n, k, a, A->data_, lda, B->data_, ldb, c, R->data_, m);
    }
    else
    {
        Cblas::gemm(ta, tb, m, n, k, a, A->data_, lda, B->data_, ldb, c, R->data_, m);
    }
}

//矩阵乘以向量，R = aAB+cR
//B和R的维度会被无视
void Matrix::mulVector(Matrix* A, Matrix* B, Matrix* R, real a /*= 1*/, real c /*= 0*/, MatrixTransType ta /*= NoTrans*/)
{
    int m = A->row_, n = A->col_;

    if (R->cuda_type_ == CUDA_GPU)
    {
        R->cuda_->cublas_->gemv(ta, m, n, a, A->data_, A->row_, B->data_, 1, c, R->data_, 1);
    }
    else
    {
        Cblas::gemv(ta, m, n, a, A->data_, A->row_, B->data_, 1, c, R->data_, 1);
    }
}

//没什么用，废弃
void Matrix::mulVector2(Matrix* A, Matrix* B, Matrix* R, real a /*= 1*/, real c /*= 0*/, MatrixTransType ta /*= NoTrans*/)
{
    int m = A->row_, n = A->col_;
    if (ta == MATRIX_TRANS) { std::swap(m, n); };

    if (R->cuda_type_ == CUDA_GPU)
    {
        for (int i = 0; i <= R->col_; i++)
        {
            R->cuda_->cublas_->gemv(ta, m, n, a, A->data_, A->row_, B->data_, 1, c, R->getDataPointer(0, i), 1);
        }
    }
    else
    {
        for (int i = 0; i <= R->col_; i++)
        {
            Cblas::gemv(ta, m, n, a, A->data_, A->row_, B->data_, 1, c, R->getDataPointer(0, i), 1);
        }
    }
}

//矩阵元素乘，B和R数据不能指向同一区域
void Matrix::elementMul(Matrix* A, Matrix* B, Matrix* R, real a, real b)
{
    if (R->cuda_type_ == CUDA_GPU)
    {
        cudnnSetOpTensorDescriptor(R->cuda_->op_tensor_desc_, CUDNN_OP_TENSOR_MUL, MYCUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN);
        //好像B不能与R相同
        if (R->data_ != B->data_)
        {
            cudnnOpTensor(R->cuda_->cudnn_handle_, R->cuda_->op_tensor_desc_,
                &a, A->tensor_desc_, A->data_, &const_real_1, B->tensor_desc_, B->data_, &b, R->tensor_desc_, R->data_);
        }
        else
        {
            cudnnOpTensor(R->cuda_->cudnn_handle_, R->cuda_->op_tensor_desc_,
                &a, B->tensor_desc_, B->data_, &const_real_1, A->tensor_desc_, A->data_, &b, R->tensor_desc_, R->data_);
        }
    }
    else
    {
        for (int i = 0; i < A->data_size_; i++)
        {
            R->data_[i] = A->data_[i] * B->data_[i] * a + R->data_[i] * b;
        }
    }
}

//矩阵加，系数为负时可以为减
void Matrix::add(Matrix* A, Matrix* B, Matrix* R, real a /*= 1*/, real b /*= 1*/)
{
    if (R->cuda_type_ == CUDA_GPU)
    {
        if (A->data_ == R->data_)
        {
            cudnnAddTensor(R->cuda_->cudnn_handle_, &b, B->tensor_desc_, B->data_, &a, A->tensor_desc_, A->data_);
        }
        else if (B->data_ == R->data_)
        {
            cudnnAddTensor(R->cuda_->cudnn_handle_, &a, A->tensor_desc_, A->data_, &b, B->tensor_desc_, B->data_);
        }
        else
        {
            //有好多种实现，geam非Blas标准
            //cuda_->cublas_->geam(Matrix_NoTrans, Matrix_NoTrans, A->row, A->col, a, A->data, A->row, b, B->data, B->row, R->data, R->row);
            cudnnSetOpTensorDescriptor(R->cuda_->op_tensor_desc_, CUDNN_OP_TENSOR_ADD, MYCUDNN_DATA_REAL, CUDNN_NOT_PROPAGATE_NAN);
            cudnnOpTensor(R->cuda_->cudnn_handle_, R->cuda_->op_tensor_desc_, &a, A->tensor_desc_, A->data_, &b, B->tensor_desc_, B->data_, &const_real_0, R->tensor_desc_, R->data_);
        }
    }
    else
    {
        for (int i = 0; i < R->data_size_; i++)
        {
            R->data_[i] = a * A->data_[i] + b * B->data_[i];
        }
    }
}

//整个矩阵点乘
real Matrix::dot(Matrix* A, Matrix* B)
{
    if (A->cuda_type_ == CUDA_GPU)
    {
        return A->cuda_->cublas_->dot(A->data_size_, A->getDataPointer(), 1, B->getDataPointer(), 1);
    }
    else
    {
        return Cblas::dot(A->data_size_, A->getDataPointer(), 1, B->getDataPointer(), 1);
    }
}

//选择矩阵的某列点乘
real Matrix::dotCol(Matrix* A, int cA, Matrix* B, int cB)
{
    if (A->cuda_type_ == CUDA_GPU)
    {
        return A->cuda_->cublas_->dot(A->row_, A->getDataPointer(0, cA), 1, B->getDataPointer(0, cA), 1);
    }
    else
    {
        return Cblas::dot(A->row_, A->getDataPointer(0, cA), 1, B->getDataPointer(0, cA), 1);
    }
}

//选择部分点乘
real Matrix::dotPart(int size, Matrix* A, real* a, int cA, real* b, int cB)
{
    if (A->cuda_type_ == CUDA_GPU)
    {
        return A->cuda_->cublas_->dot(size, a, cA, b, cB);
    }
    else
    {
        return Cblas::dot(size, a, cA, b, cB);
    }
}

//点乘，即所有元素平方和
real Matrix::dotSelf()
{
    if (cuda_type_ == CUDA_GPU)
    {
        return cuda_->cublas_->dot(data_size_, data_, 1, data_, 1);
    }
    else
    {
        return Cblas::dot(data_size_, data_, 1, data_, 1);
    }
}

//取符号
void Matrix::sign(Matrix* A, Matrix* R, real v /*= 1*/, real section /*= 1e-4*/)
{
    if (A->cuda_type_ == CUDA_GPU)
    {
        cuda_reciprocal(A->data_, R->data_, A->data_size_, v, section);
    }
    else
    {
        for (int i = 0; i < A->data_size_; i++)
        {
            if (A->data_[i] > section) { R->data_[i] = 1; continue; }
            if (A->data_[i] < -section) { R->data_[i] = -1; continue; }
            R->data_[i] = 0;
        }
    }
}

void Matrix::importData(real* v, int n)
{
    copyDataPointer(nullptr, v, this, data_, n);
    for (int i = 0; i < n; i++) { fprintf(stdout, "%f, ", v[i]); }
}

void Matrix::exportData(real* v, int n)
{

}

//求倒数，a = scale ./ a
void Matrix::reciprocal(real scale /*= 1*/)
{
    if (cuda_type_ == CUDA_GPU)
    {
        cuda_reciprocal(data_, data_, data_size_, scale, 0);
    }
    else
    {
        for (int i = 0; i < data_size_; i++)
        {
            data_[i] = scale / data_[i];
        }
    }
}

//加上一个数字，a = v + scale .* a;
void Matrix::addNumber(real v, real scale /*= 1*/)
{
    if (cuda_type_ == CUDA_GPU)
    {
        cuda_addnumber(data_, data_, data_size_, v, scale);
    }
    else
    {
        for (int i = 0; i < data_size_; i++)
        {
            data_[i] = v + scale * data_[i];
        }
    }
}

void Matrix::elementPow(Matrix* A, Matrix* R, real e, real bias)
{
    if (A->cuda_type_ == CUDA_GPU)
    {
        cuda_pow(A->data_, R->data_, A->data_size_, e, bias);
    }
    else
    {
        for (int i = 0; i < A->data_size_; i++)
        {
            R->data_[i] = pow(bias + A->data_[i], e);
        }
    }
}

void Matrix::elementDiv(Matrix* A, Matrix* B, Matrix* R, real a, real b)
{
    //b = 1;
    if (A->cuda_type_ == CUDA_GPU)
    {
        cuda_div(A->data_, B->data_, R->data_, A->data_size_, a, b);
    }
    else
    {
        for (int i = 0; i < A->data_size_; i++)
        {
            R->data_[i] = (A->data_[i] + a) / (B->data_[i] + b);
        }
    }
}


void Matrix::concatByChannel(std::vector<Matrix*> A_vector, Matrix* R)
{
    for (int n = 0; n < R->getCol(); n++)
    {
        int c_off = 0;
        for (int i = 0; i < A_vector.size(); i++)
        {
            Matrix* tmp = A_vector[i];
            copyDataPointer(tmp, tmp->getDataPointer(0, 0, 0, n), R, R->getDataPointer(0, 0, c_off, n), tmp->getRow());
            c_off += tmp->getChannel();
        }
    }

}

void Matrix::splitByChannel(Matrix* A, std::vector<Matrix*> R_vector)
{
    for (int n = 0; n < A->getCol(); n++)
    {
        int c_off = 0;
        for (int i = 0; i < R_vector.size(); i++)
        {
            Matrix* tmp = R_vector[i];
            copyDataPointer(A, A->getDataPointer(0, 0, c_off, n), tmp, tmp->getDataPointer(0, 0, 0, n), tmp->getRow());
            c_off += tmp->getChannel();
        }
    }

}
