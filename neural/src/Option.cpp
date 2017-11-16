#include "Option.h"

Option::Option()
{
    initMaps();
}

//初始化map，注意一些设置是有别名的
//所有字符小写！！！！
//第一项通常是默认项
void Option::initMaps()
{
    map_ActiveFunctionType_ =
    {
        { "none", ACTIVE_FUNCTION_NONE },
        { "sigmoid", ACTIVE_FUNCTION_SIGMOID },
        { "relu", ACTIVE_FUNCTION_RELU },
        { "tanh", ACTIVE_FUNCTION_TANH },
        { "clipped_relu", ACTIVE_FUNCTION_CLIPPED_RELU },
        { "elu", ACTIVE_FUNCTION_ELU },
        { "scale", ACTIVE_FUNCTION_SCALE },
        { "soft_max", ACTIVE_FUNCTION_SOFTMAX },
        { "soft_max_fast", ACTIVE_FUNCTION_SOFTMAX_FAST },
        { "soft_max_log", ACTIVE_FUNCTION_SOFTMAX_LOG },
        { "dropout", ACTIVE_FUNCTION_DROPOUT },
        { "recurrent", ACTIVE_FUNCTION_RECURRENT },
        { "softplus", ACTIVE_FUNCTION_SOFTPLUS },
        { "local_response_normalization", ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION },
        { "lrn", ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION },
        { "local_constrast_normalization", ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION },
        { "lcn", ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION },
        { "divisive_normalization",  ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION },
        { "dn",  ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION },
        { "batch_normalization", ACTIVE_FUNCTION_BATCH_NORMALIZATION },
        { "bn", ACTIVE_FUNCTION_BATCH_NORMALIZATION },
        { "spatial_transformer", ACTIVE_FUNCTION_SPATIAL_TRANSFORMER },
    };
    dealStringForMap(map_ActiveFunctionType_);

    map_LayerConnectionType_ =
    {
        { "none", LAYER_CONNECTION_NONE },
        { "null", LAYER_CONNECTION_NONE },
        { "fullconnect", LAYER_CONNECTION_FULLCONNECT },
        { "full", LAYER_CONNECTION_FULLCONNECT },
        { "fc", LAYER_CONNECTION_FULLCONNECT },
        { "convolution", LAYER_CONNECTION_CONVOLUTION },
        { "conv", LAYER_CONNECTION_CONVOLUTION },
        { "pooling", LAYER_CONNECTION_POOLING },
        { "pool", LAYER_CONNECTION_POOLING },
        { "direct", LAYER_CONNECTION_DIRECT },
        { "combine", LAYER_CONNECTION_COMBINE },
    };

    map_CostFunctionType_ =
    {
        { "rmse", COST_FUNCTION_RMSE },
        { "crossentropy", COST_FUNCTION_CROSS_ENTROPY },
    };

    map_PoolingType_ =
    {
        { "max", POOLING_MAX },
        { "average", POOLING_AVERAGE_NOPADDING },
        { "average_no_padding", POOLING_AVERAGE_NOPADDING },
        { "average_padding", POOLING_AVERAGE_PADDING },
    };
    dealStringForMap(map_PoolingType_);

    map_CombineType_ =
    {
        { "concat", COMBINE_CONCAT },
        { "add", COMBINE_ADD },
    };

    map_RandomFillType_ =
    {
        { "constant", RANDOM_FILL_CONSTANT },
        { "xavier", RANDOM_FILL_XAVIER },
        { "gaussian", RANDOM_FILL_GAUSSIAN },
        { "msra", RANDOM_FILL_GAUSSIAN },
    };

    map_AdjustLearnRateType_ =
    {
        { "fixed", ADJUST_LEARN_RATE_FIXED },
        { "inv", ADJUST_LEARN_RATE_INV },
        { "step", ADJUST_LEARN_RATE_STEP },
    };

    map_BatchNormalizationType_ =
    {
        { "per_active", BATCH_NORMALIZATION_PER_ACTIVATION },
        { "spatial", BATCH_NORMALIZATION_SPATIAL },
        { "auto", BATCH_NORMALIZATION_AUTO },
    };
    dealStringForMap(map_BatchNormalizationType_);

    map_RecurrentType_ =
    {
        { "relu", RECURRENT_RELU },
        { "tanh", RECURRENT_TANH },
        { "lstm", RECURRENT_LSTM },
        { "gru", RECURRENT_GRU },
    };

    map_RecurrentDirectionType_ =
    {
        { "uni", RECURRENT_DIRECTION_UNI },
        { "bi", RECURRENT_DIRECTION_BI },
    };

    map_RecurrentInputType_ =
    {
        { "linear", RECURRENT_INPUT_LINEAR },
        { "skip", RECURRENT_INPUT_SKIP },
    };

    map_RecurrentAlgoType_ =
    {
        { "standard", RECURRENT_ALGO_STANDARD },
        { "static", RECURRENT_ALGO_PERSIST_STATIC, },
        { "dynamic", RECURRENT_ALGO_PERSIST_DYNAMIC, },
    };

    map_SolverType_ =
    {
        { "sgd", SOLVER_SGD },
        { "nag", SOLVER_NAG },
        { "ada_delta", SOLVER_ADA_DELTA },
    };
    dealStringForMap(map_SolverType_);

    map_WorkModeType_ =
    {
        { "normal", WORK_MODE_NORMAL },
        { "gan", WORK_MODE_GAN },
        { "prune", WORK_MODE_PRUNE },
    };

    map_PruneType_ =
    {
        { "active", PRUNE_ACTIVE },
        { "weight", PRUNE_WEIGHT },
    };
}

void Option::loadIniFile(const std::string& filename)
{
    std::string content = convert::readStringFromFile(filename);
    loadIniString(content);
}

void Option::loadIniString(const std::string& content)
{
    ini_reader_.load(content);
    //包含文件仅载入一层
    auto filenames = convert::splitString(getString("load_ini"), ",");
    for (auto filename : filenames)
    {
        if (filename != "")
        {
            ini_reader_.load(convert::readStringFromFile(filename));
        }
    }
}

//提取字串属性，会去掉单引号，双引号
std::string Option::getStringFromSection(const std::string& section, const std::string& name, std::string v /*= ""*/)
{
    auto str = ini_reader_.Get(section, name, v);
    convert::replaceAllString(str, "\'", "");
    convert::replaceAllString(str, "\"", "");
    return str;
}

int Option::getIntFromSection2(const std::string& section, const std::string& name, int default_value)
{
    return int(getRealFromSection2(section, name, default_value));
}

//首先获取公共部分，再以公共部分为默认值获取特殊部分
real Option::getRealFromSection2(const std::string& section, const std::string& name, real default_value)
{
    real a = getReal(name, default_value);
    a = getRealFromSection(section, name, a);
    return a;
}

std::string Option::getStringFromSection2(const std::string& section, const std::string& name, std::string default_value)
{
    std::string a = getString(name, default_value);
    a = getStringFromSection(section, name, a);
    return a;
}

void Option::setOptions(std::string section, std::string name_values)
{
    setOptions(section, convert::splitString(name_values, ";"));
}

void Option::setOptions(std::string section, std::vector<std::string> name_values)
{
    for (auto name_value : name_values)
    {
        convert::replaceAllString(name_value, " ", "");
        auto p = name_value.find("=");
        if (p != std::string::npos)
        {
            auto name = name_value.substr(0, p);
            auto value = name_value.substr(p + 1);
            setOption(section, name, value);
        }
    }
}


