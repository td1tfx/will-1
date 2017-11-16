#pragma once
#include "DataPreparer.h"

class DataPreparerImage : public DataPreparer
{
private:
    int flip_transpose_ = 0;                 //旋转和翻转
    std::vector<real> d_contrast_;           //变换对比度
    std::vector<real> d_brightness_;         //变换亮度
    int d_channel_ = 0;                      //是否通道分别变换
    real d_noise_ = 0;                       //增加随机噪声

public:
    DataPreparerImage();
    virtual ~DataPreparerImage();

    void init2() override;
    void transOne(Matrix* matrix0, Matrix* matrix1, int label = 0) override;
};

