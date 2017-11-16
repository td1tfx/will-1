#include "Application.h"
#include "Brain.h"
#include "File.h"

Application::Application(const std::string& ini)
{
    ini_file_ = ini;
}

Application::~Application()
{

}

void Application::start()
{

}

void Application::stop()
{

}

void Application::test()
{
}

void Application::run()
{
    start();
    if (!File::fileExist(ini_file_))
    {
        fprintf(stderr, "%s doesn't exist!\n", ini_file_.c_str());
        return;
    }
    Brain brain;
    brain.load(ini_file_);
    if (brain.init() != 0) { return; }
    loop_ = true;
    while (loop_)
    {
        brain.run();
        loop_ = false;
    }
    stop();
}

void Application::mainLoop()
{
}

void Application::callback(void* net_pointer)
{
}
