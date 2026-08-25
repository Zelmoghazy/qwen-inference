#pragma once

#include <utils.hpp>
#include <model.hpp>

struct Data
{
    u8 *ptr;
    i64 len;
    Data(u8 *ptr, i64 len):ptr(ptr),len(len){}
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
    i8       INT8;
    u16      UINT16;
    i16      INT16;
    u32      UINT32;
    i32      INT32;
    f32      FLOAT32;
    bool     BOOL;
    u64      UINT64;
    i64      INT64;
    f64      FLOAT64;
};

struct MetaEntry
{
    u32         idx = 0;
    std::string key;
    std::string type;        
    std::string value;       
};

struct TensorEntry
{
    u32         idx = 0;
    std::string name;        
    std::string dims;        
    std::string type;        
    std::string offset;      
};

#define Consume(data, type) (type *)consume_size(data, sizeof(type))

bool parse_gguf(Data &gguf, ModelInfo &model, u64 size, u8 *file_base,std::vector<MetaEntry> &metaEntries,std::vector<TensorEntry> &topLevelTensors, std::map<uint32_t, std::vector<TensorEntry>> &blocks, std::string &architecture);