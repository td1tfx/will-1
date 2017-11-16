#pragma once
#include <string>

class Application
{
public:
    Application() {}
    Application(const std::string& ini);
    virtual ~Application();

    void start();
    void stop();
    void test();
private:
    bool loop_ = true;
    std::string ini_file_;

    void callback(void*);
public:
    void run();
    void mainLoop();
};

