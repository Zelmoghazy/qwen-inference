#include "tokenizer.hpp"
#include "gguf.hpp"
#include "engine.hpp"

#include <iostream>
#include <fstream>

extern "C"{
    #include "arena.h"
}

#define KB(n)                           (((u64)(n)) << 10)
#define MB(n)                           (((u64)(n)) << 20)
#define GB(n)                           (((u64)(n)) << 30)

void silu(const f32 *input, f32 *output, u32 size)
{
    for (u32 i = 0; i < size; i++)
    {
        f32 v = input[i];
        output[i] = v / (1.0f + expf(-v));
    }
}

void rmsnorm(const f32 *x, const f32 *weight, f32 *out, u32 n, f32 eps)
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

void rope(f32 *vec, u32 head_dim, u32 pos, f32 theta_base)
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

f32 vec_dot_f32(const f32 *a, const f32 *b, u32 n)
{
    f32 sum = 0.0f;
    for (u32 i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void softmax_f32(const f32 *input, f32 *output, u32 n)
{
    f32 max_val = input[0];
    for (u32 i = 1; i < n; i++) if (input[i] > max_val) max_val = input[i];

    f32 sum = 0.0f;
    for (u32 i = 0; i < n; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (u32 i = 0; i < n; i++) output[i] /= sum;
}

u32 argmax(const f32 *logits, u32 n)
{
    u32 best = 0;
    f32 best_val = logits[0];
    for (u32 i = 1; i < n; i++)
    {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

int main(void)
{
    Profile("Main function");

    arena_t *scratch_arena = arena_reserve(MB(500));

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
    
    for (BlockInfo &l : model.blocks)
    {
        assert(l.attn_q.dims[0]==2048 && l.attn_q.dims[1]==2048);
        assert(l.attn_k.dims[0]==2048 && l.attn_k.dims[1]==256);
        assert(l.attn_v.dims[0]==2048 && l.attn_v.dims[1]==256);
        assert(l.attn_output.dims[0]==2048 && l.attn_output.dims[1]==2048);
        assert(l.ffn_gate.dims[0]==2048 && l.ffn_up.dims[0]==2048);
        assert(l.ffn_down.dims[1]==2048);
    }

    u32 n_layers = (u32)model.blocks.size();
    u32 kv_dim   = model.cfg.attention_head_count_kv * head_dim; 
    u32 max_seq  = 2048;
    u32 ffn_dim  = model.blocks[0].ffn_gate.dims[1];
    u32 vocab    = model.token_embd.dims[1];

    f32 *kv_cache_k  = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
    f32 *kv_cache_v  = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
    
    f32 *gate        = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
    f32 *up          = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
    f32 *attn_scores = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * max_seq);
    f32 *logits      = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * vocab);

    init_tokenizer(); 
    
    std::string prompt = "Training state-of-the-art, deep neural networks is computationally expensive.";
    std::vector<int> tokens =  encode(prompt);
    std::println("{}",tokens);
    for(int token: tokens)
    {
        std::println("{}",decode_id(token));    
    }    
    u32 prompt_len = (u32)tokens.size();
    u32 max_new_tokens = 200;
    
    std::vector<f32> attn_dump; // flat buffer, we'll write it out at the end
    attn_dump.reserve((size_t)n_layers * model.cfg.attention_head_count * prompt_len * prompt_len);
    
    f32 x[2048];      

    #if 0
    for(u32 cur_pos = 0; cur_pos < prompt_len; cur_pos++)
    {
        int tok = tokens[cur_pos];
        embed_token(model.token_embd, tok, x, model.cfg.embedding_length);

        for (u32 layer_idx = 0; layer_idx < n_layers; layer_idx++)
        {
            BlockInfo &l = model.blocks[layer_idx];

            f32 normed_attn[2048];
            rmsnorm(x, (f32*)l.attn_norm.tensor_data, normed_attn, 2048, model.cfg.rms_epsilon);

            f32 q_full[2048]; 
            f32 k_full[256];  
            f32 v_full[256];

            matmul_q8_0((block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], normed_attn, q_full);
            for (u32 i = 0; i < 2048; i++) q_full[i] += ((f32*)l.attn_q_bias.tensor_data)[i];

            matmul_q8_0((block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], normed_attn, k_full);
            for (u32 i = 0; i < 256; i++) k_full[i] += ((f32*)l.attn_k_bias.tensor_data)[i];

            matmul_q8_0((block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  normed_attn, v_full);
            for (u32 i = 0; i < 256; i++) v_full[i] += ((f32*)l.attn_v_bias.tensor_data)[i];

            for (u32 h = 0; h < model.cfg.attention_head_count; h++)
                rope(q_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);
            for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++)
                rope(k_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);

            f32 *k_cache_slot = kv_cache_k + (layer_idx * max_seq + cur_pos) * kv_dim;
            f32 *v_cache_slot = kv_cache_v + (layer_idx * max_seq + cur_pos) * kv_dim;
            memcpy(k_cache_slot, k_full, sizeof(f32) * kv_dim);
            memcpy(v_cache_slot, v_full, sizeof(f32) * kv_dim);

            f32 o_full[2048]; 
            f32 scale = 1.0f / sqrtf((f32)head_dim);

            for (u32 h = 0; h < model.cfg.attention_head_count; h++)
            {
                u32 kv = h / gqa_group_size;
                f32 *qh = q_full + h*head_dim;

                for (u32 t = 0; t <= cur_pos; t++)
                {
                    f32 *kh_t = kv_cache_k + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                    attn_scores[t] = vec_dot_f32(qh, kh_t, head_dim) * scale;
                }
                softmax_f32(attn_scores, attn_scores, cur_pos + 1);

                for (u32 t = 0; t <= cur_pos; t++)
                    attn_dump.push_back(attn_scores[t]);
                for (u32 t = cur_pos + 1; t < prompt_len; t++)
                    attn_dump.push_back(0.0f);

                f32 *oh = o_full + h * head_dim;
                for (u32 i = 0; i < head_dim; i++) oh[i] = 0.0f;
                for (u32 t = 0; t <= cur_pos; t++)
                {
                    f32 *vh_t = kv_cache_v + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                    f32 w = attn_scores[t];
                    for (u32 i = 0; i < head_dim; i++) oh[i] += w * vh_t[i];
                }
            }
            f32 attn_out[2048];
            matmul_q8_0((block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], o_full, attn_out);

            f32 residual[2048];
            for (u32 i = 0; i < 2048; i++)
                residual[i] = x[i] + attn_out[i];

            f32 normed_ffn[2048];
            rmsnorm(residual, (f32*)l.ffn_norm.tensor_data, normed_ffn, 2048, model.cfg.rms_epsilon);

            matmul_q8_0((block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], normed_ffn, gate);
            matmul_q8_0((block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   normed_ffn, up);

            silu(gate, gate, ffn_dim);
            for (u32 i = 0; i < ffn_dim; i++) gate[i] *= up[i];

            f32 ffn_out[2048];
            matmul_q8_0((block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], gate, ffn_out);

            for (u32 i = 0; i < 2048; i++) x[i] = residual[i] + ffn_out[i];
        }
    }

    {
        std::ofstream f("attn_dump.bin", std::ios::binary);
        u32 header[4] = { n_layers, model.cfg.attention_head_count, prompt_len, prompt_len };
        f.write((char*)header, sizeof(header));
        f.write((char*)attn_dump.data(), attn_dump.size() * sizeof(f32));
    }
    {
        std::ofstream f("tokens.txt");
        for (u32 i = 0; i < prompt_len; i++) f << decode_id(tokens[i]) << "\n";
    }
    #endif


    #if 1

    for(u32 cur_pos = 0; cur_pos < prompt_len + max_new_tokens; cur_pos++)
    {
        int tok = (cur_pos < prompt_len) ? tokens[cur_pos] : tokens.back();
        embed_token(model.token_embd, tok, x, model.cfg.embedding_length);

        for (u32 layer_idx = 0; layer_idx < n_layers; layer_idx++)
        {
            BlockInfo &l = model.blocks[layer_idx];

            f32 normed_attn[2048];
            rmsnorm(x, (f32*)l.attn_norm.tensor_data, normed_attn, 2048, model.cfg.rms_epsilon);

            f32 q_full[2048]; 
            f32 k_full[256];  
            f32 v_full[256];

            matmul_q8_0((block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], normed_attn, q_full);
            for (u32 i = 0; i < 2048; i++) q_full[i] += ((f32*)l.attn_q_bias.tensor_data)[i];

            matmul_q8_0((block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], normed_attn, k_full);
            for (u32 i = 0; i < 256; i++) k_full[i] += ((f32*)l.attn_k_bias.tensor_data)[i];

            matmul_q8_0((block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  normed_attn, v_full);
            for (u32 i = 0; i < 256; i++) v_full[i] += ((f32*)l.attn_v_bias.tensor_data)[i];

            for (u32 h = 0; h < model.cfg.attention_head_count; h++){
                rope(q_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);
            }
            for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++){
                rope(k_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);
            }

            f32 *k_cache_slot = kv_cache_k + (layer_idx * max_seq + cur_pos) * kv_dim;
            f32 *v_cache_slot = kv_cache_v + (layer_idx * max_seq + cur_pos) * kv_dim;
            memcpy(k_cache_slot, k_full, sizeof(f32) * kv_dim);
            memcpy(v_cache_slot, v_full, sizeof(f32) * kv_dim);

            f32 o_full[2048]; 
            f32 scale = 1.0f / sqrtf((f32)head_dim);

            for (u32 h = 0; h < model.cfg.attention_head_count; h++)
            {
                u32 kv = h / gqa_group_size;
                f32 *qh = q_full + h*head_dim;

                for (u32 t = 0; t <= cur_pos; t++)
                {
                    f32 *kh_t = kv_cache_k + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                    attn_scores[t] = vec_dot_f32(qh, kh_t, head_dim) * scale;
                }
                softmax_f32(attn_scores, attn_scores, cur_pos + 1);

                f32 *oh = o_full + h * head_dim;
                for (u32 i = 0; i < head_dim; i++) oh[i] = 0.0f;
                for (u32 t = 0; t <= cur_pos; t++)
                {
                    f32 *vh_t = kv_cache_v + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                    f32 w = attn_scores[t];
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
            rmsnorm(residual, (f32*)l.ffn_norm.tensor_data, normed_ffn, 2048, model.cfg.rms_epsilon);


            matmul_q8_0((block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], normed_ffn, gate);
            matmul_q8_0((block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   normed_ffn, up);

            silu(gate, gate, ffn_dim);
            for (u32 i = 0; i < ffn_dim; i++) gate[i] *= up[i];   // gate now holds silu(gate)*up

            f32 ffn_out[2048];
            matmul_q8_0((block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], gate, ffn_out);

            for (u32 i = 0; i < 2048; i++) x[i] = residual[i] + ffn_out[i];
        }
        f32 normed_final[2048];
        if (cur_pos >= prompt_len - 1)
        {
            rmsnorm(x, (f32*)model.output_norm.tensor_data, normed_final, 2048, model.cfg.rms_epsilon);
            matmul_q8_0((block_q8_0*)model.token_embd.tensor_data, 2048, vocab, normed_final, logits);
            u32 next_token = argmax(logits, vocab);

            std::print("{}", decode_id(next_token));
            tokens.push_back(next_token);
        }
    }
    std::println("");
    #endif

    UnmapViewOfFile(file_base); 
    CloseHandle(Mapping);
    CloseHandle(File);
    return 0;
}