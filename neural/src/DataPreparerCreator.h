#pragma once
#include "DataPreparer.h"

class DataPreparerCreator : DataPreparer
{
private:
    DataPreparerCreator() {}
    virtual ~DataPreparerCreator() {}
public:
    static DataPreparer* create(Option* op, const std::string& section, int w0, int h0, int c0, int w1, int h1, int c1);
    static DataPreparer* createByReference(Option* op, const std::string& section, DataPreparer* ref);
};

