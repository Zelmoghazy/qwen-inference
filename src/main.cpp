
#include <cassert>
#include <print>
#include <string>
#include <vector>
#include "tokenizer.hpp"

#include <gguf.hpp>


struct Config
{
    u32 embedding_length;
    u32 block_count;
    u32 attention_head_count;
    u32 attention_head_count_kv;
    u32 feed_forward_length;
    u32 alignment = 32;
    f32 rms_epsilon;
    f32 rope_freq;
};

static bool ends_with(std::string_view s, std::string_view suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}


int main(void)
{
    HANDLE File = CreateFileA("C:\\Users\\zezo_\\.lmstudio\\models\\lmstudio-community"
                              "\\Qwen2.5-3B-Instruct-GGUF\\Qwen2.5-3B-Instruct-Q8_0.gguf",
                               GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE,
                               0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    LARGE_INTEGER size;
    GetFileSizeEx(File, &size);

    HANDLE Mapping = CreateFileMappingA(File, 0, PAGE_READONLY, 0, 0, 0);
    
    Data gguf;
    Config cfg;

    u8 *file_base = (u8 *)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
    gguf.ptr = file_base;
    gguf.len = static_cast<i64>(size.QuadPart);
    
    gguf_header *header = Consume(&gguf,gguf_header);
    
    for (u64 i = 0; i < header->metadata_kv_count; i++)
    {
        gguf_string key  = read_string(&gguf);
        u32        *type = Consume(&gguf, u32);

        if (!key.data || !type)
        {
            return 1;
        }

        std::string_view meta_key = sv(key);
        gguf_scalar_type value;

        bool consumed = get_scalar_value(&gguf, *type,  value);

        if(consumed)
        {
            if (ends_with(meta_key, "embedding_length"))
            {
                assert(*type == GGUF_TYPE_UINT32);
                cfg.embedding_length = value.UINT32;
            }
            else if (ends_with(meta_key, "block_count"))
            {
                assert(*type == GGUF_TYPE_UINT32);
                cfg.block_count = value.UINT32;
            }
            else if (ends_with(meta_key, "attention.head_count_kv"))
            {
                assert(*type == GGUF_TYPE_UINT32);
                cfg.attention_head_count_kv = value.UINT32;
            }
            else if (ends_with(meta_key, "attention.head_count"))
            {
                assert(*type == GGUF_TYPE_UINT32);
                cfg.attention_head_count = value.UINT32;
            }
            else if (ends_with(meta_key, "feed_forward_length"))
            {
                assert(*type == GGUF_TYPE_UINT32);
                cfg.feed_forward_length = value.UINT32;
            }
            else if (ends_with(meta_key, "attention.layer_norm_rms_epsilon"))
            {
                assert(*type == GGUF_TYPE_FLOAT32);
                cfg.rms_epsilon = value.FLOAT32;
            }
            else if (ends_with(meta_key, "rope.freq_base"))
            {
                assert(*type == GGUF_TYPE_FLOAT32);
                cfg.rope_freq = value.FLOAT32;
            }
            else if (ends_with(meta_key, "general.alignment"))
            {
                assert(*type == GGUF_TYPE_UINT32);
                cfg.alignment = value.UINT32;
            }
            std::println("{:>3}. {} = (scalar, type={})", i, meta_key, gguf_type_name(*type));
        }

        std::print("{:>3}. {} = ", i, meta_key);

        if (!print_value(&gguf, *type))
        {
            return 1;
        }
        std::println("");
    }
    std::println("");
    std::println("-- config --");
    std::println("embedding_length:       {}", cfg.embedding_length);
    std::println("block_count:            {}", cfg.block_count);
    std::println("attention_head_count:   {}", cfg.attention_head_count);
    std::println("attention_head_count_kv:{}", cfg.attention_head_count_kv);
    std::println("feed_forward_length:    {}", cfg.feed_forward_length);
    std::println("rms_epsilon:            {}", cfg.rms_epsilon);
    std::println("alignment:              {}", cfg.alignment);
    std::println("rope_freq:              {}", cfg.rope_freq);
 
    std::println("");
    std::println("-- tensors --");
 
    for (u64 i = 0; i < header->tensor_count; i++)
    {
        gguf_string name   = read_string(&gguf);
        u32        *n_dims = Consume(&gguf, u32);
        if (!name.data || !n_dims)
        {
            return 1;
        }
 
        std::vector<u64> dims(*n_dims);
        for (u32 d = 0; d < *n_dims; d++)
        {
            u64 *dim = Consume(&gguf, u64); 
            if (!dim)
            {
                return 1;
            }
            dims[d] = *dim;
        }
 
        u32 *ttype  = Consume(&gguf, u32);
        u64 *offset = Consume(&gguf, u64);
        if (!ttype || !offset)
        {
            return 1;
        }
 
        std::print("{:>3}. {:<40} dims=[", i, sv(name));
        for (u32 d = 0; d < *n_dims; d++)
        {
            if (d) std::print(",");
            std::print("{}", dims[d]);
        }
        std::println("] type={} offset={}", ggml_type_name(*ttype), *offset);
    }
    UnmapViewOfFile(file_base); 
    CloseHandle(Mapping);
    CloseHandle(File);
    return 0;
}