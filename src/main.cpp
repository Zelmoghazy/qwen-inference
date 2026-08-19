#include "tokenizer.hpp"
#include "gguf.hpp"
#include "engine.hpp"

#include <iostream>
#include <fstream>

extern "C"{
    #include "arena.h"
}

struct matmul_chunk_t 
{
    const block_q8_0 *weight;
    u32 blocks_per_row;
    u32 row_start, row_end;
    const f32 *input;
    f32 *output;
};

void matmul_chunk_job(void *data)
{
    ZoneScopedNC("Matmul Chunk", tracy::Color::Orange);
    matmul_chunk_t *c = (matmul_chunk_t*)data;
    for (u32 row = c->row_start; row < c->row_end; row++)
    {
        const block_q8_0 *row_blocks = c->weight + (u64)row * c->blocks_per_row;
        c->output[row] = dot_q8_0_f32(row_blocks, c->blocks_per_row, c->input);
    }
}

void matmul_q8_0_threaded(thread_pool_t *pool, const block_q8_0 *weight,
                           u32 n_in, u32 n_out, const f32 *input, f32 *output)
{
    ZoneScopedNC("Matrix Multiplication Multithreaded", tracy::Color::Tomato);
    assert(n_in % 32 == 0);
    u32 blocks_per_row = n_in / 32;

    u32 num_threads = (u32)pool->threads.size();

    const u32 chunks_per_thread = 8;
    u32 num_chunks = num_threads * chunks_per_thread;
    u32 rows_per_chunk = (n_out + num_chunks - 1) / num_chunks;

    static thread_local std::vector<matmul_chunk_t> chunks;
    static thread_local std::vector<job_t> jobs;

    chunks.clear();
    jobs.clear();
    
    chunks.reserve(num_chunks);
    jobs.reserve(num_chunks);

    for (u32 t = 0; t < num_chunks; t++)
    {
        u32 start = t * rows_per_chunk;
        u32 end   = (start + rows_per_chunk < n_out) ? start + rows_per_chunk : n_out;
        if (start >= end) break;

        chunks.push_back({ weight, blocks_per_row, start, end, input, output });
        jobs.push_back({ matmul_chunk_job, &chunks.back() });
    }

    threadpool_queue_jobs_batch(pool, jobs.data(), (int)jobs.size());
    threadpool_wait(pool);
}

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
        f32 freq        = 1.0f / powf(theta_base, (2.0f * i) / head_dim);
        f32 angle       = pos * freq;
        f32 c           = cosf(angle);
        f32 s           = sinf(angle);
        f32 x0          = vec[i];
        f32 x1          = vec[i + half];
        vec[i]          = x0 * c - x1 * s;
        vec[i + half]   = x0 * s + x1 * c;
    }
}

f32 vec_dot_f32(const f32 *a, const f32 *b, u32 n)
{
    f32 sum = 0.0f;
    for (u32 i = 0; i < n; i++){
        sum += a[i] * b[i];
    } 
    return sum;
}

void softmax_f32(const f32 *input, f32 *output, u32 n)
{
    // softmax is invariant to subtraction of a constant
    f32 max_val = input[0];
    for (u32 i = 1; i < n; i++){
        if (input[i] > max_val){
            max_val = input[i];
        } 
    } 
    // subtracting x_max from all x_i ensures it doesnt overflow
    f32 sum = 0.0f;
    for (u32 i = 0; i < n; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (u32 i = 0; i < n; i++){
        output[i] /= sum;
    } 
}

/* Just take the highest prob, very repetitive */
u32 argmax(const f32 *logits, u32 n)
{
    u32 best = 0;
    f32 best_val = logits[0];
    for (u32 i = 1; i < n; i++)
    {
        if (logits[i] > best_val) { 
            best_val = logits[i]; 
            best = i; 
        }
    }
    return best;
}

std::string build_chat_prompt(const std::string &system_prompt, const std::string &user_prompt)
{
    std::string p;
    p += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    p += "<|im_start|>user\n"   + user_prompt   + "<|im_end|>\n";
    p += "<|im_start|>assistant\n";   
    return p;
}


int main(void)
{
    TracySection init("Initialization");
        arena_t *scratch_arena = arena_reserve(MB(500));
        thread_pool_t *pool = threadpool_create();
        if (!pool) {
            std::println("Failed to create thread pool");
            return 1;
        }
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
        
    init.Leave();
        
    TracySection parse("Parsing GGUF");
        ModelInfo model;
        parse_gguf(gguf, model, size.QuadPart, file_base);
        
        for (BlockInfo &l : model.blocks)
        {
            assert(l.attn_q.dims[0]==2048 && l.attn_q.dims[1]==2048);
            assert(l.attn_k.dims[0]==2048 && l.attn_k.dims[1]==256);
            assert(l.attn_v.dims[0]==2048 && l.attn_v.dims[1]==256);
            assert(l.attn_output.dims[0]==2048 && l.attn_output.dims[1]==2048);
            assert(l.ffn_gate.dims[0]==2048 && l.ffn_up.dims[0]==2048);
            assert(l.ffn_down.dims[1]==2048);
        }
        
        u32 d_model         = model.cfg.embedding_length;                      // 2048
        u32 n_layers        = model.cfg.block_count;                           // 36
        u32 head_dim        = model.cfg.embedding_length / model.cfg.attention_head_count; // dk == dv
        u32 kv_dim          = model.cfg.attention_head_count_kv * head_dim;    // 256
        u32 gqa_group_size  = model.cfg.attention_head_count / model.cfg.attention_head_count_kv; // 8
        u32 max_seq         = d_model;
        u32 ffn_dim         = model.cfg.feed_forward_length;
        u32 vocab           = model.token_embd.dims[1];
        f32 scale           = 1.0f / sqrtf((f32)head_dim);  // 1 / sqrt(d_k)
        
        f32 *normed_attn    = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        f32 *q_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *k_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * kv_dim);    
        f32 *v_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * kv_dim);

        f32 *kv_cache_k     = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
        f32 *kv_cache_v     = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
        
        f32 *o_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        f32 *attn_scores    = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * max_seq);
        f32 *gate           = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
        f32 *up             = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
        f32 *logits         = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * vocab);
        
        
        f32 *attn_out       = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        f32 *residual       = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *normed_ffn     = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *ffn_out        = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *normed_final   = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    

        f32 *x              = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
    parse.Leave();
        
    TracySection tokenization("Prompt Tokenization");
        init_tokenizer(); 
        std::string prompt = build_chat_prompt(
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            "Give me a Rust program the implements a very simple shell."
        );
        std::vector<int> tokens = encode(prompt);
        std::println("{}",tokens);

        for(int token: tokens){
            std::println("{}",decode_id(token));    
        }    

        u32 prompt_len = (u32)tokens.size();
        u32 max_new_tokens = 1000;
    tokenization.Leave();
    

    #if 1
    for(u32 cur_pos = 0; cur_pos < prompt_len + max_new_tokens; cur_pos++)
    {
        int tok = (cur_pos < prompt_len) ? tokens[cur_pos] : tokens.back();
        embed_token(model.token_embd, tok, x, model.cfg.embedding_length);

        for (u32 layer_idx = 0; layer_idx < n_layers; layer_idx++)
        {
            BlockInfo &l = model.blocks[layer_idx];

            /* ------------------------------------- */
            TracySection RMSNorm1("RMSNorm1");

                rmsnorm(x, (f32*)l.attn_norm.tensor_data, normed_attn, d_model, model.cfg.rms_epsilon);
            
            RMSNorm1.Leave();
            /* ------------------------------------- */


            /* ------------------------------------- */
            TracySection attention("Attention");
                
                matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], normed_attn, q_full);
                for (u32 i = 0; i < d_model; i++) {
                    q_full[i] += ((f32*)l.attn_q_bias.tensor_data)[i];
                }

                mat_vec_mul_q8_0((block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], normed_attn, k_full);
                for (u32 i = 0; i < kv_dim; i++){
                    k_full[i] += ((f32*)l.attn_k_bias.tensor_data)[i];
                }

                mat_vec_mul_q8_0((block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  normed_attn, v_full);
                for (u32 i = 0; i < kv_dim; i++){
                    v_full[i] += ((f32*)l.attn_v_bias.tensor_data)[i];
                }
                
                {
                    ZoneScopedN("RoPE");
                    for (u32 h = 0; h < model.cfg.attention_head_count; h++){
                        rope(q_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);
                    }
                    for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++){
                        rope(k_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);
                    }
                }

                f32 *k_cache_slot = kv_cache_k + (layer_idx * max_seq + cur_pos) * kv_dim;
                f32 *v_cache_slot = kv_cache_v + (layer_idx * max_seq + cur_pos) * kv_dim;
                memcpy(k_cache_slot, k_full, sizeof(f32) * kv_dim);
                memcpy(v_cache_slot, v_full, sizeof(f32) * kv_dim);
            
                {
                    ZoneScopedN("Attention Heads");
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
                        for (u32 i = 0; i < head_dim; i++){
                            oh[i] = 0.0f;
                        } 
                        for (u32 t = 0; t <= cur_pos; t++)
                        {
                            f32 *vh_t = kv_cache_v + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                            f32 w = attn_scores[t];
                            for (u32 i = 0; i < head_dim; i++){
                                oh[i] += w * vh_t[i];
                            }
                        }
                    }
                }
                
                matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], o_full, attn_out);
                
                // update embeddings
                for (u32 i = 0; i < d_model; i++){
                    residual[i] = x[i] + attn_out[i]; 
                }
            attention.Leave();
            /* ------------------------------------- */
            
            /* ------------------------------------- */
            TracySection ffn("FFN");

                rmsnorm(residual, (f32*)l.ffn_norm.tensor_data, normed_ffn, d_model, model.cfg.rms_epsilon);

                matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], normed_ffn, gate);
                matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   normed_ffn, up);

                silu(gate, gate, ffn_dim);
                for (u32 i = 0; i < ffn_dim; i++){
                    gate[i] *= up[i];   // gate now holds silu(gate)*up
                }

                matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], gate, ffn_out);

                for (u32 i = 0; i < d_model; i++){
                    x[i] = residual[i] + ffn_out[i];
                }

            ffn.Leave();
            /* ------------------------------------- */

        }
    
        /* ------------------------------------- */
        TracySection post_norm("Final RMSNorm");
            if (cur_pos >= prompt_len - 1)
            {
                rmsnorm(x, (f32*)model.output_norm.tensor_data, normed_final, d_model, model.cfg.rms_epsilon);
                matmul_q8_0_threaded(pool,(block_q8_0*)model.token_embd.tensor_data, d_model, vocab, normed_final, logits);
                u32 next_token = argmax(logits, vocab);
                if (next_token == 151645 || next_token == 151643){
                    break;
                }
                std::print("{}", decode_id(next_token));
                tokens.push_back(next_token);
            }
        post_norm.Leave();
        /* ------------------------------------- */

        FrameMarkNamed("Token");
    }
    std::println("");
    threadpool_destroy(pool);
    #endif

    #if 0
    for(u32 cur_pos = 0; cur_pos < prompt_len; cur_pos++)
    {
        int tok = tokens[cur_pos];
        embed_token(model.token_embd, tok, x, model.cfg.embedding_length);

        for (u32 layer_idx = 0; layer_idx < n_layers; layer_idx++)
        {
            BlockInfo &l = model.blocks[layer_idx];

            f32 normed_attn[d_model];
            rmsnorm(x, (f32*)l.attn_norm.tensor_data, normed_attn, d_model, model.cfg.rms_epsilon);

            f32 q_full[d_model]; 
            f32 k_full[256];  
            f32 v_full[256];

            matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], normed_attn, q_full);
            for (u32 i = 0; i < d_model; i++) q_full[i] += ((f32*)l.attn_q_bias.tensor_data)[i];

            matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], normed_attn, k_full);
            for (u32 i = 0; i < 256; i++) k_full[i] += ((f32*)l.attn_k_bias.tensor_data)[i];

            matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  normed_attn, v_full);
            for (u32 i = 0; i < 256; i++) v_full[i] += ((f32*)l.attn_v_bias.tensor_data)[i];

            for (u32 h = 0; h < model.cfg.attention_head_count; h++)
                rope(q_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);
            for (u32 h = 0; h < model.cfg.attention_head_count_kv; h++)
                rope(k_full + h*head_dim, head_dim, cur_pos, model.cfg.rope_freq);

            f32 *k_cache_slot = kv_cache_k + (layer_idx * max_seq + cur_pos) * kv_dim;
            f32 *v_cache_slot = kv_cache_v + (layer_idx * max_seq + cur_pos) * kv_dim;
            memcpy(k_cache_slot, k_full, sizeof(f32) * kv_dim);
            memcpy(v_cache_slot, v_full, sizeof(f32) * kv_dim);

            f32 o_full[d_model]; 
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
            f32 attn_out[d_model];
            matmul_q8_0_threaded(pool,(block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], o_full, attn_out);

            f32 residual[d_model];
            for (u32 i = 0; i < d_model; i++)
                residual[i] = x[i] + attn_out[i];

            f32 normed_ffn[d_model];
            rmsnorm(residual, (f32*)l.ffn_norm.tensor_data, normed_ffn, d_model, model.cfg.rms_epsilon);

            matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_gate.tensor_data, l.ffn_gate.dims[0], l.ffn_gate.dims[1], normed_ffn, gate);
            matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_up.tensor_data,   l.ffn_up.dims[0],   l.ffn_up.dims[1],   normed_ffn, up);

            silu(gate, gate, ffn_dim);
            for (u32 i = 0; i < ffn_dim; i++) gate[i] *= up[i];

            f32 ffn_out[d_model];
            matmul_q8_0_threaded(pool,(block_q8_0*)l.ffn_down.tensor_data, l.ffn_down.dims[0], l.ffn_down.dims[1], gate, ffn_out);

            for (u32 i = 0; i < d_model; i++) x[i] = residual[i] + ffn_out[i];
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

    UnmapViewOfFile(file_base); 
    CloseHandle(Mapping);
    CloseHandle(File);
    return 0;
}