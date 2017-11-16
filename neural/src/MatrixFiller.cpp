#include "MatrixFiller.h"

MatrixFiller::MatrixFiller()
{
}


MatrixFiller::~MatrixFiller()
{
}

void MatrixFiller::fill(Matrix* m, RandomFillType random_type, int in, int out)
{
    Random<real> random_generator;
    random_generator.set_seed();
    real a = 0, b = 0;

    switch (random_type)
    {
    case RANDOM_FILL_CONSTANT:
        m->initData(0);
        return;
        break;
    case RANDOM_FILL_XAVIER:
        random_generator.set_random_type(RANDOM_UNIFORM);
        a = sqrt(6.0 / (in + out));
        random_generator.set_parameter(-a, a);
        //LOG("Xavier, %d, %d, %f\n", prev_layer->out_total, out_total, a);
        break;
    case RANDOM_FILL_GAUSSIAN:
        random_generator.set_random_type(RANDOM_NORMAL);
        //LOG("Gaussian distribution\n");
        break;
    case RANDOM_FILL_MSRA:
        random_generator.set_random_type(RANDOM_NORMAL);
        a = sqrt(2.0 / in);
        random_generator.set_parameter(0, a);
        //LOG("Gaussian distribution\n");
        break;
    default:
        break;
    }
    m->initRandom(&random_generator);
}
