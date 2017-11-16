#pragma once
#include "Neural.h"
#include "Matrix.h"
#include "types.h"

class MatrixFiller
{
public:
    MatrixFiller();
    virtual ~MatrixFiller();

    static void fill(Matrix* m, RandomFillType random_type, int in, int out);
};

