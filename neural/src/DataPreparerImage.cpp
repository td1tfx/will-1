#include "DataPreparerImage.h"
#include <time.h>
#include "File.h"
#include "others/libconvert.h"
#include <time.h>
#include <iostream>
#include "Timer.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <atomic>

DataPreparerImage::DataPreparerImage()
{
}

DataPreparerImage::~DataPreparerImage()
{
}

void DataPreparerImage::init2()
{
    setLog(option_->getIntFromSection(section_, "output_log", 0));

    LOG("Options for image processing %s:\n", section_.c_str());

    OPTION_GET_INT(flip_transpose_);
    OPTION_GET_NUMVECTOR(d_contrast_, 0, 2);
    OPTION_GET_NUMVECTOR(d_brightness_, 0, 2);

    OPTION_GET_INT(d_channel_);
    OPTION_GET_REAL(d_noise_);

    setLog(1);
    //LOG("Options for image processing %s end\n\n", section_.c_str());
}

//变换一张图
void DataPreparerImage::transOne(Matrix* matrix0, Matrix* matrix1, int label /*= 0*/)
{
    Matrix::copyData(matrix0, matrix1);
    if (flip_transpose_ != 0)
    {
        //flip -1, 0, 1
        matrix1->filp(floor(rand_.rand() * 4) - 1);
        //transpose -1, 0, 1
        matrix1->transpose(floor(rand_.rand() * 2) - 1);
    }

    auto temp = new Matrix(matrix1->getWidth(), matrix1->getHeight(), matrix1->getChannel(), 1, MATRIX_DATA_OUTSIDE, matrix1->getCudaType());
    auto temp_one_channel = new Matrix(matrix1->getWidth(), matrix1->getHeight(), 1, 1, MATRIX_DATA_OUTSIDE, matrix1->getCudaType());
    bool need_limit = false;
    temp->shareData(matrix1->getDataPointer());
    //噪点
    if (d_noise_ != 0 && temp->getCudaType() == CUDA_CPU)
    {
        need_limit = true;
        for (int i = 0; i < temp->getDataSize(); i++)
        {
            temp->getData(i, 0) += rand_fast() * d_noise_ * 2 - d_noise_;
        }
    }
    //亮度
    if (d_brightness_[1] > d_brightness_[0])
    {
        need_limit = true;
        if (d_channel_ == 0)
        {
            temp->addNumber(d_brightness_[0] + (d_brightness_[1] - d_brightness_[0]) * rand_fast());
        }
        else
        {
            for (int i = 0; i < matrix1->getChannel(); i++)
            {
                temp_one_channel->shareData(matrix1->getDataPointer(0, 0, i, 0));
                temp_one_channel->addNumber(d_brightness_[0] + (d_brightness_[1] - d_brightness_[0]) * rand_fast());
            }
        }
    }
    //对比度
    if (d_contrast_[1] > d_contrast_[0])
    {
        need_limit = true;
        if (d_channel_ == 0)
        {
            temp->scale(1 + d_contrast_[0] + (d_contrast_[1] - d_contrast_[0]) * rand_fast());
        }
        else
        {
            for (int i = 0; i < matrix1->getChannel(); i++)
            {
                temp_one_channel->shareData(matrix1->getDataPointer(0, 0, i, 0));
                temp_one_channel->scale(1 + d_contrast_[0] + (d_contrast_[1] - d_contrast_[0]) * rand_fast());
            }
        }
    }
    //printf("%f ", temp->getData(64, 64, 0, 0));
    if (need_limit) { temp->sectionLimit(0, 1); }
    //printf("%f\n", temp->getData(64, 64, 0, 0));
    safe_delete({ &temp, &temp_one_channel });
}

