#include "tokenizer.hpp"
#include "gguf.hpp"
#include "engine.hpp"


void silu(const f32 *input, f32 *output, u32 size)
{
    for (u32 i = 0; i < size; i++)
    {
        f32 v = input[i];
        output[i] = v / (1.0f + expf(-v));
    }
}

void rmsnorm(const f32 *x, const f32 *weight, f32 *out, u32 n, f32 eps = 1e-6f)
{
    f32 ss = 0.0f;
    for (u32 i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    f32 scale = 1.0f / sqrtf(ss / n + eps);
    for (u32 i = 0; i < n; i++) {
        out[i] = x[i] * scale * weight[i];
    }
}

void rope(f32 *vec, u32 head_dim, u32 pos, f32 theta_base = 1000000.0f)
{
    u32 half = head_dim / 2;
    for (u32 i = 0; i < half; i++)
    {
        f32 freq = 1.0f / powf(theta_base, (2.0f * i) / head_dim);
        f32 angle = pos * freq;
        f32 c = cosf(angle), s = sinf(angle);
        f32 x0 = vec[i], x1 = vec[i + half];
        vec[i]        = x0 * c - x1 * s;
        vec[i + half] = x0 * s + x1 * c;
    }
}

int main(void)
{
    Profile("Main function");

    HANDLE File = CreateFileA("C:\\Users\\zezo_\\.lmstudio\\models\\lmstudio-community"
                              "\\Qwen2.5-3B-Instruct-GGUF\\Qwen2.5-3B-Instruct-Q8_0.gguf",
                               GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE,
                               0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    LARGE_INTEGER size;
    GetFileSizeEx(File, &size);

    HANDLE Mapping = CreateFileMappingA(File, 0, PAGE_READONLY, 0, 0, 0);
    
    Data gguf;
    ModelInfo model;

    u8 *file_base = (u8 *)MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
    gguf.ptr = file_base;
    gguf.len = static_cast<i64>(size.QuadPart);
    
    parse_gguf(gguf, model, size.QuadPart, file_base);


    u32 head_dim = model.cfg.embedding_length / model.cfg.attention_head_count;
    assert(head_dim == 128);

    u32 gqa_group_size = model.cfg.attention_head_count / model.cfg.attention_head_count_kv;
    assert(gqa_group_size == 8);

    u32 n_layers = (u32)model.blocks.size();
    u32 kv_dim = 256; // attention_head_count_kv * head_dim
    u32 max_seq = 2048;
    f32 *kv_cache_k = (f32*)malloc(sizeof(f32) * n_layers * max_seq * kv_dim);
    f32 *kv_cache_v = (f32*)malloc(sizeof(f32) * n_layers * max_seq * kv_dim);

    u32 ffn_dim = model.blocks[0].ffn_gate.dims[1];
    f32 *gate = (f32*)malloc(sizeof(f32) * ffn_dim);
    f32 *up   = (f32*)malloc(sizeof(f32) * ffn_dim);
    f32 *scores = (f32*)malloc(sizeof(f32) * max_seq);

    std::string prompt = "Hello, world!";
    std::vector<int> tokens =  encode(prompt);    

    for(u32 cur_pos = 0; cur_pos < tokens.size(); cur_pos++)
    {
        f32 x[2048];      
        embed_token(model.token_embd, tokens[cur_pos], x, model.cfg.embedding_length);

        for (u32 layer_idx = 0; layer_idx < n_layers; layer_idx++)
        {
            BlockInfo &l = model.blocks[layer_idx];

            f32 normed_attn[2048];
            rmsnorm(x, (f32*)l.attn_norm.tensor_data, normed_attn, 2048);

            f32 q_full[2048]; 
            f32 k_full[256];  
            f32 v_full[256];

            matmul_q8_0((block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], x, q_full);
            matmul_q8_0((block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], x, k_full);
            matmul_q8_0((block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  x, v_full);

            for (u32 h = 0; h < model.cfg.attention_head_count; h++)
                rope(q_full + h*head_dim, head_dim, cur_pos);
            for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++)
                rope(k_full + h*head_dim, head_dim, cur_pos);

            f32 *k_cache_slot = kv_cache_k + (layer_idx * max_seq + cur_pos) * kv_dim;
            f32 *v_cache_slot = kv_cache_v + (layer_idx * max_seq + cur_pos) * kv_dim;
            memcpy(k_cache_slot, k_full, sizeof(f32) * kv_dim);
            memcpy(v_cache_slot, v_full, sizeof(f32) * kv_dim);

            f32 o_full[2048]; 

            for (u32 h = 0; h < model.cfg.attention_head_count; h++)
            {
                u32 kv = h / gqa_group_size;
                f32 *qh = q_full + h*head_dim;
                f32 scale = 1.0f / sqrtf((f32)head_dim);

                for (u32 t = 0; t <= cur_pos; t++)
                {
                    f32 *kh_t = kv_cache_k + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                    scores[t] = vec_dot_f32(qh, kh_t, head_dim) * scale;
                }
                softmax_f32(scores, scores, cur_pos + 1);

                f32 *oh = o_full + h * head_dim;
                for (u32 i = 0; i < head_dim; i++) oh[i] = 0.0f;
                for (u32 t = 0; t <= cur_pos; t++)
                {
                    f32 *vh_t = kv_cache_v + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                    f32 w = scores[t];
                    for (u32 i = 0; i < head_dim; i++) oh[i] += w * vh_t[i];
                }
            }
            f32 attn_out[2048];
            matmul_q8_0((block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], o_full, attn_out);

            f32 residual[2048];
            for (u32 i = 0; i < 2048; i++){
                residual[i] = x[i] + attn_out[i]; 
            }


            f32 normed_ffn[2048];
            rmsnorm(residual, (f32*)l.ffn_norm.tensor_data, normed_ffn, 2048);


            matmul_q8_0((block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], residual, gate);
            matmul_q8_0((block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   residual, up);

            silu(gate, gate, ffn_dim);
            for (u32 i = 0; i < ffn_dim; i++) gate[i] *= up[i];   // gate now holds silu(gate)*up

            f32 ffn_out[2048];
            matmul_q8_0((block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], gate, ffn_out);

            for (u32 i = 0; i < 2048; i++) x[i] = residual[i] + ffn_out[i];
        }
    }


    UnmapViewOfFile(file_base); 
    CloseHandle(Mapping);
    CloseHandle(File);
    return 0;
}