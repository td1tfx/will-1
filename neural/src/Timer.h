#pragma once
#include <string>
#include <chrono>
#include <ctime>

//计时器类
class Timer
{
private:
    std::chrono::time_point<std::chrono::system_clock> t0_, t1_;
    bool running_ = false;
public:
    Timer() { start(); }
    ~Timer() {}

    //以字符串返回当前时间
    static std::string getNowAsString()
    {
        auto t = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(t);
        std::string str = std::ctime(&time);
        return str;
    }

    void start()
    {
        running_ = true;
        t0_ = std::chrono::system_clock::now();
    }

    void stop()
    {
        running_ = false;
        t1_ = std::chrono::system_clock::now();
    }

    double getElapsedTime()
    {
        if (running_)
        {
            t1_ = std::chrono::system_clock::now();
        }
        auto s = std::chrono::duration_cast<std::chrono::duration<double>>(t1_ - t0_);
        return s.count();
    }
};