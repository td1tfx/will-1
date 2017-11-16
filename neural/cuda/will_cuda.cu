
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.hpp"
#undef __CUDA_INTERNAL_COMPILATION__
#include <stdio.h>
#include "will_cuda.h"

#define blockMax 1024

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)

inline int blockNum(unsigned int size) { return (size + blockMax - 1) / blockMax; }

inline int getError(const char* content)
{
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "%s Kernel launch failed: %s\n", content, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

#define CUDA_FUNCTION22(name, function)\
    __global__ void name##kernel(real* p1, real* p2, unsigned int size, real a1, real a2)\
    {\
        int i = cal_i();\
        if (i < size) { function }\
    }\
    int name(real* p1, real* p2, unsigned int size, real a1, real a2)\
    {\
        name##kernel <<< blockNum(size), blockMax >>> (p1, p2, size, a1, a2);\
        return getError(#name);\
    }

#define CUDA_FUNCTION32(name, function)\
    __global__ void name##kernel(real* p1, real* p2, real* p3, unsigned int size, real a1, real a2)\
    {\
        int i = cal_i();\
        if (i < size) { function }\
    }\
    int name(real* p1, real* p2, real* p3, unsigned int size, real a1, real a2)\
    {\
        name##kernel <<< blockNum(size), blockMax >>> (p1, p2, p3, size, a1, a2);\
        return getError(#name);\
    }

#define CUDA_FUNCTION42(name, function)\
    __global__ void name##kernel(real* p1, real* p2, real* p3, real* p4, unsigned int size, real a1, real a2)\
    {\
        int i = cal_i();\
        if (i < size) { function }\
    }\
    int name(real* p1, real* p2, real* p3, real* p4, unsigned int size, real a1, real a2)\
    {\
        name##kernel <<< blockNum(size), blockMax >>> (p1, p2, p3, p4, size, a1, a2);\
        return getError(#name);\
    }


CUDA_FUNCTION22(cuda_reciprocal, { p2[i] = a1 / (p1[i] + a2); })
CUDA_FUNCTION22(cuda_addnumber, { p2[i] = a1 + p1[i] * a2; })
CUDA_FUNCTION22(cuda_pow, { p2[i] = pow(p1[i] + a2, a1); })
CUDA_FUNCTION22(cuda_sparse,
{
    p2[i] = ((1 - a1) / (1 - p1[i]) - a1 / p1[i]) * a2;
})

CUDA_FUNCTION22(cuda_sign,
{
    if (p1[i] > a2) { p2[i] = a1; return; }
    if (p1[i] < -a2) { p2[i] = -a1; return; }
    p2[i] = 0;
})

CUDA_FUNCTION32(cuda_add, { p3[i] = p1[i] * a1 + p2[i] * a2; })
CUDA_FUNCTION32(cuda_mul, { p3[i] = p1[i] * p2[i] * a1 + p3[i] * a2; })
CUDA_FUNCTION32(cuda_div,
{
    p3[i] = (p1[i] + a1) / (p2[i] + a2);
})
CUDA_FUNCTION32(cuda_sectionlimit,
{
    if (p3 != p1) { p3[i] = p1[i]; }
    if (p3[i] < a1) { p3[i] = a1; }
    if (p3[i] > a2) { p3[i] = a2; }
})
CUDA_FUNCTION32(cuda_ada_update,
{
    real& rou = a1;
    real& epsilon = a2;
    p2[i] = p2[i] * rou + p3[i] * p3[i] * (1 - rou);
    p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
    p1[i] = p1[i] * rou + p3[i] * p3[i] * (1 - rou);
});

CUDA_FUNCTION42(cuda_adaDelta_update,
{
    real& rou = a1;
    real& epsilon = a2;
    p1[i] = p1[i] * rou + p3[i] * p3[i] * (1 - rou);
    p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
    p2[i] = p2[i] * rou + p4[i] * p4[i] * (1 - rou);
});


