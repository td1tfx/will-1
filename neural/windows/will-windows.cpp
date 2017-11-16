#include <cstdio>
#include "Application.h"
#include "cmdline.h"

int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add("train", '\0', "train a net");
  
    //train and detect
    cmd.add<std::string>("config", 'c', "config file (ini format) to describe the net structure", true, "will.ini");

    cmd.parse_check(argc, argv);

    auto ini_file = cmd.get<std::string>("config");

    if (cmd.exist("train"))
    {
        Application will(ini_file);
        will.run();

    }
    else if (cmd.exist("test"))
    {
        Application will;
        will.test();
    }
    else
    {
        fprintf(stdout, "Please use -? or --help to check the options.\n");
    }

    return 0;
}
