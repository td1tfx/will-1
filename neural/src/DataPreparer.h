#pragma once
#include "Option.h"
#include "DataGroup.h"
#include "Neural.h"

class DataPreparer : public Neural
{
public:
    friend class DataPreparerCreator;
public:
    DataPreparer();
    virtual ~DataPreparer();
protected:
    void init();
    virtual void init2() {}
    int getTrainDataForceGroup();
public:
    virtual void fillData(DataGroup& data) {}
    void transData(DataGroup& data0, DataGroup& data1, const std::vector<int>& fill_queue);
    virtual void transOne(Matrix* matrix0, Matrix* matrix1, int label = 0);
    virtual void showData(DataGroup& data, int number) {}
protected:
    std::string getSection() { return section_; }

private:
    //对train数据进行变换的参数
    std::vector<int> train_queue_origin_;
    int trained_in_origin_ = 0;

public:
    void reload();
    void shuffleQueue(std::vector<int>& train_queue);
    void prepareData(int epoch, DataGroup& origin, DataGroup& data);
    void prepareData(int epoch, DataGroup& origin) { prepareData(epoch, origin, origin); }

    void read(DataGroup& train, DataGroup& test);
    void readTxt(const std::string& filename, DataGroup& data);
    void readBin(const std::string& file_txt, const std::string& file_bin, DataGroup& data);

    void setNullData(DataGroup& data);
    void resizeDataGroup(DataGroup& data);

protected:
    int fill_ = 0;
    int trans_ = 0;
    std::string section_ = "data_preparer";
    Option* option_;
    int aem_ = 0;
    Random<double> rand_;
    real rand_fast() { return 1.0 * rand() / RAND_MAX; }

    //图的尺寸
    int w0_ = 1, w1_ = 1;
    int h0_ = 1, h1_ = 1;
    int c0_ = 1, c1_ = 1;
    int n_ = 0;
    void setWHC(int w0, int h0, int c0, int w1, int h1, int c1);

protected:
    std::vector<std::string> message_;
public:
    std::string getMessage(int i);

};

#define SECTION section_
#define OPTION_GET_INT(a) do { a = option_->getIntFromSection(SECTION, #a, a); LOG("%s\b: %d\n", #a, a); } while (0)
#define OPTION_GET_INT2(a, v) do { a = option_->getIntFromSection(SECTION, #a, v); LOG("%s\b: %d\n", #a, a); } while (0)
#define OPTION_GET_NUMVECTOR(a, v, n) do { convert::findNumbers(option_->getStringFromSection(SECTION, #a), &a); fillNumVector(a, v, n); LOG("%s\b: ", #a); printNumVector(a); } while (0)
#define OPTION_GET_REAL(a) do { a = option_->getRealFromSection(SECTION, #a, a); LOG("%s\b: %g\n", #a, a); } while (0)
#define OPTION_GET_REAL2(a, v) do { a = option_->getRealFromSection(SECTION, #a, v); LOG("%s\b: %g\n", #a, a); } while (0)
#define OPTION_GET_STRING(a) do { a = option_->getRealFromSection(SECTION, #a, ""); } while (0)
