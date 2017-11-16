#include "DataPreparerCreator.h"
#include "DataPreparerImage.h"
#include "DataPreparerTxt.h"
#include "mythapi.h"

DataPreparer* DataPreparerCreator::create(Option* op, const std::string& section, int w0, int h0, int c0, int w1, int h1, int c1)
{
    DataPreparer* dp = nullptr;

    if (dp == nullptr)
    {
        auto mode = op->getStringFromSection(section, "mode", "image");

        if (mode == "image")
        {
            dp = new DataPreparerImage();
            LOG("Create default image data preparer\n");
        }
        else if (mode == "txt")
        {
            dp = new DataPreparerTxt();
            LOG("Create default txt data preparer\n");
        }
        else
        {
            dp = new DataPreparer();
            LOG("Create default data preparer\n");
        }
    }

    dp->option_ = op;
    dp->section_ = section;
    dp->setWHC(w0, h0, c0, w1, h1, c1);
    dp->init();

    return dp;
}

DataPreparer* DataPreparerCreator::createByReference(Option* op, const std::string& section, DataPreparer* ref)
{
    return create(op, section, ref->w0_, ref->h0_, ref->c0_, ref->w1_, ref->h1_, ref->c1_);
}


