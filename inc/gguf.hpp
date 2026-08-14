#pragma once


#include <utils.hpp>
#include <model.hpp>


struct Data
{
    u8 *ptr;
    i64 len;
};

/*
    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */

#pragma pack(push, 1)
struct GGUFHeader
{
    u32 magic;              // @Note: Little Endian
    u32 version;
    u64 tensor_count;
    u64 metadata_kv_count; 
};
#pragma pack(pop)
static_assert(sizeof(GGUFHeader) == 24, "GGUFHeader should be 24 bytes");

enum GGUFType : u32
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

union GGUFScalarType
{
    u8       UINT8;
    int8_t   INT8;
    uint16_t UINT16;
    int16_t  INT16;
    u32      UINT32;
    int32_t  INT32;
    f32      FLOAT32;
    bool     BOOL;
    u64      UINT64;
    i64      INT64;
    f64      FLOAT64;
};

#define Consume(data, type) (type *)consume_size(data, sizeof(type))

void* consume_size(Data* d, u32 size);
bool print_value(Data *d, u32 type, int depth=0, u64 preview = 8);
bool skip_value(Data *d, u32 type, int depth = 0);
std::string_view read_gguf_string(Data *d);
const char* ggml_type_name(u32 t);
bool get_scalar_value(Data *d, u32 type, GGUFScalarType &out);
const char* gguf_type_name(u32 t);
void print_scalar_value(u32 idx, std::string_view &meta_key, u32 type, GGUFScalarType value);
bool parse_gguf(Data &gguf, ModelInfo &model, u64 size, u8 *file_base);
