#pragma once
#include <string>
#include <vector>
#include <stdarg.h>

//神经层，神经网的基类，包含一些常用功能
class Neural
{
public:
    Neural();
    virtual ~Neural();

    template <class T> static void safe_delete(T*& pointer)
    {
        if (pointer) { delete pointer; }
        pointer = nullptr;
    }
    template <class T> static void safe_delete(std::vector<T*>& pointer_v)
    {
        for (auto& pointer : pointer_v)
        {
            safe_delete(pointer);
        }
    }
    template <class T> static void safe_delete(std::initializer_list<T**> pointer_v)
    {
        for (auto& pointer : pointer_v)
        {
            safe_delete(*pointer);
        }
    }
    static int LOG(const char* format, ...);

public:
    template <class T> static void printNumVector(std::vector<T> v)
    {
        for (auto value : v)
        {
            LOG("%g, ", double(value));
        }
        LOG("\b\b \n");
    }
    template <class T> static void fillNumVector(std::vector<T>& v, double value, int size)
    {
        for (int i = v.size(); i < size; i++)
        {
            v.push_back(T(value));
        }
    }

private:
    static int log_;
public:
    static void setLog(int log) { log_ = log; }
};

