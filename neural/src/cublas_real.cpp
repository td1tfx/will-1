#include "cublas_real.h"

#ifndef _NO_CUDA
#ifdef STATIC_BLAS
cublasHandle_t Cublas::handle_;
#endif
#endif
