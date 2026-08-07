
#include <print>
#include <string>
#include <vector>
#include "tokenizer.hpp"

#include <gguf.hpp>

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
 
        std::print("{:>3}. {} = ", i, sv(key));
        if (!print_value(&gguf, *type))
        {
            return 1;
        }
        std::println("");
    }
 
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