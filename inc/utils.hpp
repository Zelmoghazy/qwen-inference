#pragma once

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN 
    #define VC_EXTRALEAN
    #define NOMINMAX
    #include <windows.h>
#endif

#include <cstdint>
#include <cassert>
#include <print>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>

#define PROFILE 1
#define LOG 0

#if PROFILE
    #include "tracy/tracy/Tracy.hpp"
    #define Profile(s) ZoneScopedN(s)
#else
    #define Profile(s)
#endif

typedef uint8_t u8;
typedef int8_t i8;
typedef uint16_t u16;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
typedef float f32;
typedef double f64;

#define ALIGN_UP(x, y)                  ((((x) + (y) - 1) / (y)) * (y))
