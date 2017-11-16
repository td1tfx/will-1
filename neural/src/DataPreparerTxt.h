#pragma once
#include "DataPreparer.h"
#include <string>

class DataPreparerTxt : public DataPreparer
{
private:
    int input_;
    int output_;    
    std::string content_;

public:
    DataPreparerTxt();
    virtual ~DataPreparerTxt();

    void init2() override;
    void fillData(DataGroup& data) override;

private:
    float getContent(int i);
};

