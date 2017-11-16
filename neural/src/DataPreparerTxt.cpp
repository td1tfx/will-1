#include "DataPreparerTxt.h"
#include "File.h"

DataPreparerTxt::DataPreparerTxt()
{
}

DataPreparerTxt::~DataPreparerTxt()
{
}

void DataPreparerTxt::init2()
{
    input_ = w0_ * h0_ * c0_;
    output_ = w1_ * h1_ * c1_;
    std::string filename = option_->getStringFromSection(section_, "file", "file.txt");
    content_ = convert::readStringFromFile(filename);
}

void DataPreparerTxt::fillData(DataGroup& data)
{
    rand_.set_seed();

    for (int index = 0; index < data.getNumber(); index++)
    {
        int r = rand_.rand() * (content_.size() / 2 - input_ - output_);
        for (int i = 0; i < input_; i++)
        {
            data.X()->getData(i, index) = getContent(r + i);
        }
        for (int i = 0; i < output_; i++)
        {
            data.Y()->getData(i, index) = getContent(r + i + input_);
        }
    }
}

float DataPreparerTxt::getContent(int i)
{
    auto p = (uint16_t*)(&content_[i * 2]);
    return (*p) / 65536.0;
}
