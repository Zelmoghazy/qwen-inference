#include "gguf.hpp"

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

/* 
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
 */
const char* ggml_type_name(u32 t)
{
    switch (t)
    {
        case 0:  
            return "f32";
        case 1:  
            return "f16";
        case 2:  
            return "q4_0";
        case 3:  
            return "q4_1";
        case 6:  
            return "q5_0";
        case 7:  
            return "q5_1";
        case 8:  
            return "q8_0";
        case 9:  
            return "q8_1";
        case 10: 
            return "q2_k";
        case 11: 
            return "q3_k";
        case 12: 
            return "q4_k";
        case 13: 
            return "q5_k";
        case 14: 
            return "q6_k";
        default: 
            return "?";
    }
}

const char* gguf_type_name(u32 t)
{
    switch (t)
    {
        case GGUF_TYPE_UINT8:   
            return "uint8";
        case GGUF_TYPE_INT8:    
            return "int8";
        case GGUF_TYPE_UINT16:  
            return "uint16";
        case GGUF_TYPE_INT16:   
            return "int16";
        case GGUF_TYPE_UINT32:  
            return "uint32";
        case GGUF_TYPE_INT32:   
            return "int32";
        case GGUF_TYPE_FLOAT32: 
            return "float32";
        case GGUF_TYPE_BOOL:    
            return "bool";
        case GGUF_TYPE_STRING:  
            return "string";
        case GGUF_TYPE_ARRAY:   
            return "array";
        case GGUF_TYPE_UINT64:  
            return "uint64";
        case GGUF_TYPE_INT64:   
            return "int64";
        case GGUF_TYPE_FLOAT64: 
            return "float64";
        default:                
            return "unknown";
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
/*
    struct gguf_string_t 
    {
        // The length of the string, in bytes.
        uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        char string[len];
    }; 
 */
std::string_view read_gguf_string(Data *d)
{
    u64 *len = Consume(d, u64);
    if (!len) {
        return std::string_view{};
    }
    char *data  = (char *)consume_size(d, *len);
    return data ? std::string_view(data, *len) : std::string_view{};
}

static bool skip_scalar(Data *d, u32 type)
{
    if (type == GGUF_TYPE_STRING)
    {
        std::string_view s = read_gguf_string(d);
        return s.data() != nullptr || s.length() == 0;
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

bool get_scalar_value(Data *d, u32 type, GGUFScalarType &out)
{
    switch (type)
    {
    case GGUF_TYPE_UINT8: {
        u8 *v = Consume(d, u8);
        if (!v){
            return false;
        }
        out.UINT8 = *v;
        return true;
    }
    case GGUF_TYPE_INT8: {
        i8 *v = Consume(d, int8_t);
        if (!v){
            return false;
        }
        out.INT8 = *v;
        return true;
    }
    case GGUF_TYPE_UINT16: {
        u16 *v = Consume(d, uint16_t);
        if (!v){
            return false;
        }
        out.UINT16 = *v;
        return true;
    }
    case GGUF_TYPE_INT16: {
        i16 *v = Consume(d, int16_t);
        if (!v){
            return false;
        }
        out.INT16 = *v;
        return true;
    }
    case GGUF_TYPE_UINT32: {
        u32 *v = Consume(d, u32);
        if (!v){
            return false;
        }
        out.UINT32 = *v;
        return true;
    }
    case GGUF_TYPE_INT32: {
        i32 *v = Consume(d, int32_t);
        if (!v){
            return false;
        }
        out.INT32 = *v;
        return true;
    }
    case GGUF_TYPE_FLOAT32: {
        f32 *v = Consume(d, f32);
        if (!v){
            return false;
        }
        out.FLOAT32 = *v;
        return true;
    }
    case GGUF_TYPE_BOOL: {
        u8 *v = Consume(d, u8);
        if (!v){
            return false;
        }
        out.BOOL = *v;
        return true;
    }
    case GGUF_TYPE_UINT64: {
        u64 *v = Consume(d, u64);
        if (!v){
            return false;
        }
        out.UINT64 = (f64) *v;
        return true;
    }
    case GGUF_TYPE_INT64: {
        i64 *v = Consume(d, i64);
        if (!v){
            return false;
        }
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

static std::string scalar_to_str(u32 type, GGUFScalarType value)
{
    switch (type)
    {
        case GGUF_TYPE_UINT8:   return std::to_string(value.UINT8);
        case GGUF_TYPE_INT8:    return std::to_string(value.INT8);
        case GGUF_TYPE_UINT16:  return std::to_string(value.UINT16);
        case GGUF_TYPE_INT16:   return std::to_string(value.INT16);
        case GGUF_TYPE_UINT32:  return std::to_string(value.UINT32);
        case GGUF_TYPE_INT32:   return std::to_string(value.INT32);
        case GGUF_TYPE_FLOAT32: return std::to_string(value.FLOAT32);
        case GGUF_TYPE_BOOL:    return value.BOOL ? "true" : "false";
        case GGUF_TYPE_UINT64:  return std::to_string(value.UINT64);
        case GGUF_TYPE_INT64:   return std::to_string(value.INT64);
        case GGUF_TYPE_FLOAT64: return std::to_string(value.FLOAT64);
        default:                return "?";
    }
}

static std::string value_to_string(Data *d, u32 type, int depth, u64 preview = 8)
{
    if (type == GGUF_TYPE_STRING)
    {
        return std::string(read_gguf_string(d));
    }

    if (type != GGUF_TYPE_ARRAY)
    {
        GGUFScalarType value;
        get_scalar_value(d, type, value);
        return scalar_to_str(type, value);
    }

    if (depth > 8) return "";

    u32 *elem_type = Consume(d, u32);
    u64 *count     = Consume(d, u64);
    if (!elem_type || !count) return "";

    const u64 to_show = std::min(*count, preview);

    std::string out = "[";
    for (u64 i = 0; i < to_show; i++)
    {
        if (i) out += ", ";
        out += value_to_string(d, *elem_type, depth + 1,  preview);
    }

    for (u64 i = to_show; i < *count; i++)
    {
        if (!skip_value(d, *elem_type, depth + 1))
            break; 
    }

    if (*count > to_show) out += ", ...";
    out += "]";
    return out;
}

bool parse_u32(std::string_view idx_str, u32& idx)
{
    if (idx_str.empty()){
        return false;
    }

    auto [ptr, ec] = std::from_chars(
        idx_str.data(),
        idx_str.data() + idx_str.size(),
        idx
    );

    return (ec == std::errc{}) &&
           (ptr == idx_str.data()) + idx_str.size();
}

static bool ends_with(std::string_view s, std::string_view suffix)
{
    return (s.size() >= suffix.size()) &&
           (s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0);
}

static bool parse_blk_name(std::string_view name, u32& layer_idx, std::string_view& rest, u32 max_block_count)
{
    if (!name.starts_with("blk.")){
        return false;
    }

    name.remove_prefix(4);

    size_t dot = name.find('.');
    if (dot == std::string_view::npos){
        return false;
    }

    std::string_view idx_str = name.substr(0, dot);

    if (!parse_u32(idx_str, layer_idx)){
        return false;
    }

    if (layer_idx >= max_block_count){
        std::println("tensor {} -> layer {} but block_count is {}", name, layer_idx, max_block_count);
        return false;
    }

    rest = name.substr(dot + 1);

    return true;
}


bool parse_gguf(Data &gguf, ModelInfo &model, u64 size, u8 *file_base,               
                std::vector<MetaEntry> &metaEntries,
                std::vector<TensorEntry> &topLevelTensors,
                std::map<uint32_t, std::vector<TensorEntry>> &blocks,
                std::string &architecture)
{
    ZoneScopedN("GGUF Parser");

    GGUFHeader *header = Consume(&gguf, GGUFHeader);
    
    /*
        struct gguf_metadata_kv_t 
        {
            // The key of the metadata. It is a standard GGUF string, with the following caveats:
            // - It must be a valid ASCII string.
            // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
            // - It must be at most 2^16-1/65535 bytes long.
            // Any keys that do not follow these rules are invalid.

            gguf_string_t key;

            // The type of the value.
            // Must be one of the `gguf_metadata_value_type` values.

            gguf_metadata_value_type value_type;

            // The value.
            gguf_metadata_value_t value;
        }; 
    */
    for (u64 i = 0; i < header->metadata_kv_count; i++)
    {
        std::string_view key         = read_gguf_string(&gguf);
        u32             *value_type  = Consume(&gguf, u32);

        GGUFScalarType value;
        bool consumed = get_scalar_value(&gguf, *value_type,  value);

        if(consumed)
        {
            if (ends_with(key, "embedding_length"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.embedding_length = value.UINT32;
            }
            else if (ends_with(key, "block_count"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.block_count = value.UINT32;
            }
            else if (ends_with(key, "attention.head_count_kv"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.attention_head_count_kv = value.UINT32;
            }
            else if (ends_with(key, "attention.head_count"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.attention_head_count = value.UINT32;
            }
            else if (ends_with(key, "feed_forward_length"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.feed_forward_length = value.UINT32;
            }
            else if (ends_with(key, "attention.layer_norm_rms_epsilon"))
            {
                assert(*value_type == GGUF_TYPE_FLOAT32);
                model.cfg.rms_epsilon = value.FLOAT32;
            }
            else if (ends_with(key, "rope.freq_base"))
            {
                assert(*value_type == GGUF_TYPE_FLOAT32);
                model.cfg.rope_freq = value.FLOAT32;
            }
            else if (ends_with(key, "general.alignment"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.alignment = value.UINT32;
            }
            else if (ends_with(key, "context_length"))
            {
                assert(*value_type == GGUF_TYPE_UINT32);
                model.cfg.context_length = value.UINT32;
            }
            metaEntries.push_back({(uint32_t)i, std::string(key), gguf_type_name(*value_type), scalar_to_str(*value_type, value)});
        }
        else
        {
            std::string val_str;

            if (*value_type == GGUF_TYPE_STRING)
            {
                std::string_view value = read_gguf_string(&gguf);
                if (key == "general.name"){
                    model.name = std::string(value);
                }
                if (key == "general.architecture"){
                    architecture = std::string(value);
                }
                val_str = std::string(value);
            }
            else
            {
                val_str = value_to_string(&gguf, *value_type, 0);
            }

            metaEntries.push_back({(uint32_t)i, std::string(key), gguf_type_name(*value_type), val_str});

        }
    }
    model.blocks.resize(model.cfg.block_count);
    /*
        struct gguf_tensor_info_t 
        {
            // The name of the tensor. It is a standard GGUF string, with the caveat that
            // it must be at most 64 bytes long.

            gguf_string_t name;

            // The number of dimensions in the tensor.
            // Currently at most 4, but this may change in the future.

            uint32_t n_dimensions;

            // The dimensions of the tensor.

            uint64_t dimensions[n_dimensions];

            // The type of the tensor.

            ggml_type type;

            // The offset of the tensor's data in this file in bytes.
            //
            // This offset is relative to `tensor_data`, not to the start
            // of the file, to make it easier for writers to write the file.
            // Readers should consider exposing this offset relative to the
            // file to make it easier to read the data.
            //
            // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.

            uint64_t offset;
        }; 
    */
    for (u64 i = 0; i < header->tensor_count; i++)
    {
        std::string_view name    = read_gguf_string(&gguf);
        u32              *n_dims = Consume(&gguf, u32);

        u64 dims[4]={0};
        for (u32 d = 0; d < *n_dims; d++){
            dims[d] = *Consume(&gguf, u64); 
        }
 
        u32 *tensor_type = Consume(&gguf, u32);
        u64 *offset = Consume(&gguf, u64);

        TensorInfo *slot = nullptr;
        u32 layer_idx; 
        std::string_view rest;
        bool is_block_tensor = parse_blk_name(name, layer_idx, rest, model.cfg.block_count);

        if (is_block_tensor)
        {
            if (rest == "attn_norm.weight"){
                slot = &model.blocks[layer_idx].attn_norm;
            } 
            else if (rest == "attn_q.weight"){
                slot = &model.blocks[layer_idx].attn_q;
            }
            else if (rest == "attn_q.bias"){
                slot = &model.blocks[layer_idx].attn_q_bias;
            }
            else if (rest == "attn_k.weight"){
                slot = &model.blocks[layer_idx].attn_k;
            }      
            else if (rest == "attn_k.bias"){
                slot = &model.blocks[layer_idx].attn_k_bias;
            }
            else if (rest == "attn_v.weight"){
                slot = &model.blocks[layer_idx].attn_v;
            }
            else if (rest == "attn_v.bias"){
                slot = &model.blocks[layer_idx].attn_v_bias;
            }
            else if (rest == "attn_output.weight"){
                slot = &model.blocks[layer_idx].attn_output;
            }
            else if (rest == "ffn_norm.weight"){
                slot = &model.blocks[layer_idx].ffn_norm;
            }
            else if (rest == "ffn_gate.weight"){
                slot = &model.blocks[layer_idx].ffn_gate;
            }
            else if (rest == "ffn_up.weight"){
                slot = &model.blocks[layer_idx].ffn_up;
            }
            else if (rest == "ffn_down.weight"){
                slot = &model.blocks[layer_idx].ffn_down;
            }
        }
        else if (name == "token_embd.weight") 
        {
            slot = &model.token_embd;
        }
        else if (name == "output_norm.weight") 
        {
            slot = &model.output_norm;
        }
        else if (name == "output.weight")      
        {
            slot = &model.output;
        }

        if (slot)
        {
            memcpy(slot->dims, dims, sizeof(dims));
            slot->n_dims  = (u8)*n_dims;
            slot->type    = *tensor_type;
            slot->offset  = *offset;
            slot->present = true;
        }
        else
        {
            std::println("unrecognized tensor: {}", name);
        }
        std::string dims_str = "[";
        for (u32 d = 0; d < *n_dims; d++)
        {
            if (d) dims_str += ",";
            dims_str += std::to_string(dims[d]);
        }
        dims_str += "]";

        TensorEntry te{(uint32_t)i, std::string(name), dims_str, ggml_type_name(*tensor_type), std::to_string(*offset)};

        if (is_block_tensor)
        {
            blocks[layer_idx].push_back(std::move(te));
        }
        else
        {
            topLevelTensors.push_back(std::move(te));
        }
    }

    u64 consumed   = size - (u64)gguf.len;
    u64 data_start = ALIGN_UP(consumed, model.cfg.alignment);
    u8 *data_base  = file_base + data_start;

    auto set_data_ptr = [&](TensorInfo &t) { if (t.present) t.tensor_data = data_base + t.offset; };

    set_data_ptr(model.token_embd);
    set_data_ptr(model.output_norm);
    set_data_ptr(model.output);

    for (BlockInfo &l : model.blocks)
    {
        set_data_ptr(l.attn_norm);  
        set_data_ptr(l.attn_q);  
        set_data_ptr(l.attn_q_bias);

        set_data_ptr(l.attn_k);     
        set_data_ptr(l.attn_k_bias);
        set_data_ptr(l.attn_v);     
        set_data_ptr(l.attn_v_bias);
        set_data_ptr(l.attn_output);
        set_data_ptr(l.ffn_norm);
        set_data_ptr(l.ffn_gate); 
        set_data_ptr(l.ffn_up); 
        set_data_ptr(l.ffn_down);
    }

    if(model.name == "Qwen2.5 3B Instruct")
    {
        for (BlockInfo &l : model.blocks)
        {
            assert(l.attn_q.dims[0]==2048 && l.attn_q.dims[1]==2048);
            assert(l.attn_k.dims[0]==2048 && l.attn_k.dims[1]==256);
            assert(l.attn_v.dims[0]==2048 && l.attn_v.dims[1]==256);
            assert(l.attn_output.dims[0]==2048 && l.attn_output.dims[1]==2048);
            assert(l.ffn_gate.dims[0]==2048 && l.ffn_up.dims[0]==2048);
            assert(l.ffn_down.dims[1]==2048);
        }
    }

    return true;
}

static u64 tensor_elem_count(const TensorInfo& t)
{
    if (!t.present) return 0;
    u64 n = 1;
    for (u32 d = 0; d < t.n_dims; d++){
        n *= t.dims[d];
    }
    return n;
}

static u64 total_param_count(const ModelInfo& model)
{
    u64 total = tensor_elem_count(model.token_embd)
              + tensor_elem_count(model.output_norm)
              + tensor_elem_count(model.output);

    for (const BlockInfo& blk : model.blocks)
    {
        total += tensor_elem_count(blk.attn_norm)
               + tensor_elem_count(blk.attn_q)      + tensor_elem_count(blk.attn_q_bias)
               + tensor_elem_count(blk.attn_k)      + tensor_elem_count(blk.attn_k_bias)
               + tensor_elem_count(blk.attn_v)      + tensor_elem_count(blk.attn_v_bias)
               + tensor_elem_count(blk.attn_output)
               + tensor_elem_count(blk.ffn_norm)
               + tensor_elem_count(blk.ffn_gate)
               + tensor_elem_count(blk.ffn_up)
               + tensor_elem_count(blk.ffn_down);
    }
    return total;
}

static std::string format_param_count(u64 n)
{
    double v = (double)n;
    const char* suffix = "";
    if      (n >= 1'000'000'000ull) { v /= 1e9; suffix = "B"; }
    else if (n >= 1'000'000ull)     { v /= 1e6; suffix = "M"; }
    else if (n >= 1'000ull)         { v /= 1e3; suffix = "K"; }
    else return std::to_string(n);

    std::ostringstream oss;
    oss.precision(v < 10.0 ? 2 : 1);
    oss << std::fixed << v << suffix;
    return oss.str();
}

static std::string format_thousands(u64 n)
{
    std::string s = std::to_string(n);
    for (i64 i = (i64)s.size() - 3; i > 0; i -= 3) s.insert(i, ",");
    return s;
}

static std::string replace_all(std::string s, std::string_view from, std::string_view to)
{
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos)
    {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

bool export_architecture_svg(const ModelInfo& model,
                              const std::string& template_path,
                              const std::string& output_path)
{
    std::ifstream in(template_path, std::ios::binary);
    if (!in) { std::println("could not open svg template: {}", template_path); return false; }

    std::stringstream buf;
    buf << in.rdbuf();
    std::string svg = buf.str();

    const u32 n_vocab  = model.token_embd.present ? (u32)model.token_embd.dims[1] : 0;
    const u64 n_params = total_param_count(model);

    struct Replacement { const char* key; std::string value; };
    Replacement subs[] = {
        { "{{model_name}}",  model.name.empty() ? "unnamed model" : model.name },
        { "{{param_count}}", format_param_count(n_params) },
        { "{{n_vocab}}",     format_thousands(n_vocab) },
        { "{{n_embd}}",      format_thousands(model.cfg.embedding_length) },
        { "{{n_ff}}",        format_thousands(model.cfg.feed_forward_length) },
        { "{{n_heads}}",     std::to_string(model.cfg.attention_head_count) },
        { "{{n_kv_heads}}",  std::to_string(model.cfg.attention_head_count_kv) },
        { "{{n_ctx}}",       format_thousands(model.cfg.context_length) },
        { "{{n_layers}}",    std::to_string(model.cfg.block_count) },
    };

    for (const Replacement& r : subs) svg = replace_all(svg, r.key, r.value);

    std::ofstream out(output_path, std::ios::binary);
    if (!out) { std::println("could not write svg output: {}", output_path); return false; }
    out << svg;

    std::println("wrote architecture card -> {}", output_path);
    return true;
}