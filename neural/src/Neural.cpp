#include "Neural.h"

int Neural::log_ = 1;

Neural::Neural()
{
}


Neural::~Neural()
{
}

int Neural::LOG(const char* format, ...)
{
    if (log_ == 0) { return 0; }
    va_list arg_ptr;
    va_start(arg_ptr, format);
    char s[1024];
    int n = vsnprintf(s, 1024, format, arg_ptr);
    va_end(arg_ptr);
    fprintf(stdout, "%s", s);
    return n;
}
