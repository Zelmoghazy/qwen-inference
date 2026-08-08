
#include <gguf.hpp>

long long get_file_size(const char *FileName) 
{
    WIN32_FILE_ATTRIBUTE_DATA fad;

    if (!GetFileAttributesExA(FileName, GetFileExInfoStandard, &fad)) {
        return -1;
    }

    LARGE_INTEGER size;
    size.HighPart = fad.nFileSizeHigh;
    size.LowPart  = fad.nFileSizeLow;

    return size.QuadPart;
}

#define Consume(data, type) (type *)consume_size(data, sizeof(type))

void* consume_size(Data* d, u32 size)
{
    void *result = nullptr;

    if(d->len >= size)
    {
        result = d->ptr;
        d->ptr = (u8*)d->ptr + size;
        d->len -= size;
    }
    else
    {
        d->len = 0;
    }

    return result;
}

const char* ggml_type_name(u32 t)
{
    switch (t)
    {
        case 0:  return "f32";
        case 1:  return "f16";
        case 2:  return "q4_0";
        case 3:  return "q4_1";
        case 6:  return "q5_0";
        case 7:  return "q5_1";
        case 8:  return "q8_0";
        case 9:  return "q8_1";
        case 10: return "q2_k";
        case 11: return "q3_k";
        case 12: return "q4_k";
        case 13: return "q5_k";
        case 14: return "q6_k";
        default: return "?";
    }
}

const char* gguf_type_name(u32 t)
{
    switch (t)
    {
        case GGUF_TYPE_UINT8:   return "uint8";
        case GGUF_TYPE_INT8:    return "int8";
        case GGUF_TYPE_UINT16:  return "uint16";
        case GGUF_TYPE_INT16:   return "int16";
        case GGUF_TYPE_UINT32:  return "uint32";
        case GGUF_TYPE_INT32:   return "int32";
        case GGUF_TYPE_FLOAT32: return "float32";
        case GGUF_TYPE_BOOL:    return "bool";
        case GGUF_TYPE_STRING:  return "string";
        case GGUF_TYPE_ARRAY:   return "array";
        case GGUF_TYPE_UINT64:  return "uint64";
        case GGUF_TYPE_INT64:   return "int64";
        case GGUF_TYPE_FLOAT64: return "float64";
        default:                return "unknown";
    }
}

static u64 gguf_type_size(u32 t)
{
    switch (t)
    {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

gguf_string read_string(Data *d)
{
    gguf_string s;
    u64 *len = Consume(d, u64);
    if (!len) {
        return s;
    }
 
    s.len  = *len;
    s.data = (const char *)consume_size(d, s.len);
    return s;
}

std::string_view sv(const gguf_string &s)
{
    return s.data ? std::string_view(s.data, s.len) : std::string_view{};
}

static bool skip_scalar(Data *d, u32 type)
{
    if (type == GGUF_TYPE_STRING)
    {
        gguf_string s = read_string(d);
        return s.data != nullptr || s.len == 0;
    }
    u64 size = gguf_type_size(type);
    if (size == 0) {
        return false; 
    }
    return consume_size(d, size) != nullptr;
}
 
static bool skip_value(Data *d, u32 type, int depth)
{
    if (type != GGUF_TYPE_ARRAY){
        return skip_scalar(d, type);
    }
 
    if (depth > 8) {
        return false; 
    }
 
    u32 *elem_type = Consume(d, u32);
    u64 *count     = Consume(d, u64);

    if (!elem_type || !count) {
        return false;
    }
 
    for (u64 i = 0; i < *count; i++){
        if (!skip_value(d, *elem_type, depth + 1)) {
            return false;
        }
    }
 
    return true;
}

static bool print_scalar(Data *d, u32 type)
{
    switch (type)
    {
    case GGUF_TYPE_UINT8: {
        u8 *v = Consume(d, u8);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_INT8: {
        int8_t *v = Consume(d, int8_t);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_UINT16: {
        uint16_t *v = Consume(d, uint16_t);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_INT16: {
        int16_t *v = Consume(d, int16_t);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_UINT32: {
        u32 *v = Consume(d, u32);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_INT32: {
        int32_t *v = Consume(d, int32_t);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_FLOAT32: {
        f32 *v = Consume(d, f32);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_BOOL: {
        u8 *v = Consume(d, u8);
        if (!v)
            return false;
        std::print("{}", *v ? "true" : "false");
        return true;
    }
    case GGUF_TYPE_UINT64: {
        u64 *v = Consume(d, u64);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_INT64: {
        i64 *v = Consume(d, i64);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_FLOAT64: {
        f64 *v = Consume(d, f64);
        if (!v)
            return false;
        std::print("{}", *v);
        return true;
    }
    case GGUF_TYPE_STRING: {
        gguf_string s = read_string(d);
        if (!s.data && s.len)
            return false;
        std::print("\"{}\"", sv(s));
        return true;
    }
    default:
        return false;
    }
}

bool get_scalar_value(Data *d, u32 type, gguf_scalar_type &out)
{
    switch (type)
    {
    case GGUF_TYPE_UINT8: {
        u8 *v = Consume(d, u8);
        if (!v)
            return false;
        out.UINT8 = *v;
        return true;
    }
    case GGUF_TYPE_INT8: {
        int8_t *v = Consume(d, int8_t);
        if (!v)
            return false;
        out.INT8 = *v;
        return true;
    }
    case GGUF_TYPE_UINT16: {
        uint16_t *v = Consume(d, uint16_t);
        if (!v)
            return false;
        out.UINT16 = *v;
        return true;
    }
    case GGUF_TYPE_INT16: {
        int16_t *v = Consume(d, int16_t);
        if (!v)
            return false;
        out.INT16 = *v;
        return true;
    }
    case GGUF_TYPE_UINT32: {
        u32 *v = Consume(d, u32);
        if (!v)
            return false;
        out.UINT32 = *v;
        return true;
    }
    case GGUF_TYPE_INT32: {
        int32_t *v = Consume(d, int32_t);
        if (!v)
            return false;
        out.INT32 = *v;
        return true;
    }
    case GGUF_TYPE_FLOAT32: {
        f32 *v = Consume(d, f32);
        if (!v)
            return false;
        out.FLOAT32 = *v;
        return true;
    }
    case GGUF_TYPE_BOOL: {
        u8 *v = Consume(d, u8);
        if (!v)
            return false;
        out.BOOL = *v;
        return true;
    }
    case GGUF_TYPE_UINT64: {
        u64 *v = Consume(d, u64);
        if (!v)
            return false;
        out.UINT64 = (f64) *v;
        return true;
    }
    case GGUF_TYPE_INT64: {
        i64 *v = Consume(d, i64);
        if (!v)
            return false;
        out.INT64 = (f64) *v;
        return true;
    }
    case GGUF_TYPE_FLOAT64: {
        f64 *v = Consume(d, f64);
        if (!v)
            return false;
        out.FLOAT64 = *v;
        return true;
    }
    default:
        return false; 
    }
}

bool print_value(Data *d, u32 type, int depth, u64 preview)
{
    if (type != GGUF_TYPE_ARRAY){
        return print_scalar(d, type);
    }
 
    if (depth > 8) {
        return false;
    }
 
    u32 *elem_type = Consume(d, u32);
    u64 *count     = Consume(d, u64);

    if (!elem_type || !count) {
        return false;
    }
 
    std::print("[{} x {}] (", *count, gguf_type_name(*elem_type));
 
    const u64 to_show  = *count < preview ? *count : preview;
 
    for (u64 i = 0; i < to_show; i++)
    {
        if (i) std::print(", ");
        if (!print_value(d, *elem_type, depth + 1)) return false;
    }

    for (u64 i = to_show; i < *count; i++) {
        if (!skip_value(d, *elem_type, depth + 1)) return false;
    }
 
    if (*count > to_show) std::print(", ...");
    std::print(")");
    
    return true;
}

