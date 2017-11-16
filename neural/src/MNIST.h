#pragma once
#include "DataGroup.h"

class MNIST
{
public:
    MNIST();
    virtual ~MNIST();
    const int label_ = 10;

private:
    void getDataSize(const std::string& file_image, int* w, int* h, int* n);
    void readLabelFile(const std::string& filename, real* y_data);
    void readImageFile(const std::string& filename, real* x_data);
    void readData(const std::string& file_label, const std::string& file_image, DataGroup& data);

public:
    void load(DataGroup& train, DataGroup& test, std::string path = "mnist", int flag = 0);
};

