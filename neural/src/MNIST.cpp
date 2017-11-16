#include "MNIST.h"
#include "File.h"

#define _SAVE_MNIST0

#ifdef _SAVE_MNIST
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

MNIST::MNIST()
{
}


MNIST::~MNIST()
{
}

void MNIST::getDataSize(const std::string& file_image, int* w, int* h, int* n)
{
    auto content = File::readFile(file_image.c_str(), 16);
    File::reverse(content + 4, 4);
    File::reverse(content + 8, 4);
    File::reverse(content + 12, 4);
    if (n) { *n = *(int*)(content + 4); }
    if (w) { *w = *(int*)(content + 8); }
    if (h) { *h = *(int*)(content + 12); }
    File::deleteBuffer(content);
}

void MNIST::readLabelFile(const std::string& filename, real* y_data)
{
    int s = 10;
    auto content = File::readFile(filename.c_str());
    File::reverse(content + 4, 4);
    int n = *(int*)(content + 4);
    memset(y_data, 0, sizeof(real) * n * s);
    for (int i = 0; i < n; i++)
    {
        int pos = *(content + 8 + i);
        y_data[i * s + pos % s] = 1;
    }
    File::deleteBuffer(content);
}

void MNIST::readImageFile(const std::string& filename, real* x_data)
{
    auto content = File::readFile(filename.c_str());
    File::reverse(content + 4, 4);
    File::reverse(content + 8, 4);
    File::reverse(content + 12, 4);
    int n = *(int*)(content + 4);
    int w = *(int*)(content + 8);
    int h = *(int*)(content + 12);
    int size = n * w * h;
    memset(x_data, 0, sizeof(real)*size);
    for (int i = 0; i < size; i++)
    {
        auto v = *(content + 16 + i);
        x_data[i] = v / real(255.0);
    }
#ifdef _SAVE_MNIST
    cv::Mat A10 = cv::Mat::zeros(280, 280, CV_8UC1);
    int k = 0;
    for (int c = 0; c < 100; c++)
    {
        cv::Mat A1 = A10(cv::Rect(c / 10 * 28, c % 10 * 28, 28, 28));
        for (int i = 0; i < 784; i++)
        {
            A1.at<uint8_t>(i) = x_data[k++] * 255;
        }
    }
    cv::imwrite("mnist100.png", A10);
#endif
    File::deleteBuffer(content);
}

void MNIST::readData(const std::string& file_label, const std::string& file_image, DataGroup& data)
{
    int w, h, n;
    getDataSize(file_image, &w, &h, &n);
    data.clear();
    data.setX(new Matrix(w, h, 1, n, MATRIX_DATA_INSIDE, CUDA_CPU));
    data.setY(new Matrix(1, 1, label_, n, MATRIX_DATA_INSIDE, CUDA_CPU));
    //train.createA();
    readLabelFile(file_label, data.Y()->getDataPointer());
    readImageFile(file_image, data.X()->getDataPointer());
}

//flag: 0 - train and test, 1 - train only, 2 - test only
void MNIST::load(DataGroup& train, DataGroup& test, std::string path /*= "mnist"*/, int flag /*= 0*/)
{
    if (path.back() != '/' || path.back() != '\\') { path += '/'; }

    std::string train_label = path + "train-labels.idx1-ubyte";
    std::string train_image = path + "train-images.idx3-ubyte";
    std::string test_label = path + "t10k-labels.idx1-ubyte";
    std::string test_image = path + "t10k-images.idx3-ubyte";

    if (flag == 0 || flag == 1)
    {
        fprintf(stdout, "Loading MNIST train data...");
        readData(train_label, train_image, train);
        fprintf(stdout, "done\n");
    }
    if (flag == 0 || flag == 2)
    {
        fprintf(stdout, "Loading MNIST test data...");
        readData(test_label, test_image, test);
        fprintf(stdout, "done\n");
    }
}