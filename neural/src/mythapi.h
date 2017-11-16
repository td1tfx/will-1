#pragma once

#ifdef _WIN32
#define MYTHAPI _stdcall
#define HBAPI __declspec (dllexport)
#else
#define MYTHAPI
#define HBAPI
#endif

