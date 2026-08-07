#pragma once

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN 
    #define VC_EXTRALEAN
    #include <windows.h>
#endif

#include <cstdint>
#include <print>

typedef uint8_t u8;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
typedef float f32;
typedef double f64;

struct Data
{
    u8 *ptr;
    i64 len;
};

#pragma pack(push, 1)
struct gguf_header
{
    u32 magic;
    u32 version;
    u64 tensor_count;
    u64 metadata_kv_count; 
};
#pragma pack(pop)

struct gguf_string
{
    u64         len  = 0;
    const char *data = nullptr;
};

enum gguf_type : u32
{
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};
#define Consume(data, type) (type *)consume_size(data, sizeof(type))

void* consume_size(Data* d, u32 size);
bool print_value(Data *d, u32 type, int depth=0, u64 preview = 8);
bool skip_value(Data *d, u32 type, int depth = 0);
gguf_string read_string(Data *d);
std::string_view sv(const gguf_string &s);
const char* ggml_type_name(u32 t);
