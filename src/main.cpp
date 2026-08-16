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

#ifdef _WIN32
    typedef HANDLE thread_handle_t;
    typedef DWORD (WINAPI *thread_func_t)(LPVOID);
    typedef LPVOID thread_func_param_t;
    typedef DWORD thread_func_ret_t;
    typedef CONDITION_VARIABLE cond_handle_t;
    typedef CRITICAL_SECTION mutex_handle_t;
    typedef HANDLE pipe_handle;
    typedef HANDLE event_handle;
    typedef volatile LONG atomic_int_t;
#else
    typedef pthread_t thread_handle_t;
    typedef void* (*thread_func_t)(void*);
    typedef void* thread_func_param_t;
    typedef void* thread_func_ret_t;
    typedef pthread_mutex_t mutex_handle_t;
    typedef pthread_cond_t cond_handle_t;
    typedef int pipe_handle;
    typedef struct {
        pthread_mutex_t mutex;
        pthread_cond_t cond;
        bool signaled;
    }event_handle;
    typedef atomic_int atomic_int_t;
#endif

typedef void (*job_func_t)(void* data);

typedef struct 
{
    job_func_t func;
    void* data;
}job_t;

typedef struct 
{
    std::vector<thread_handle_t> threads;
    std::vector<job_t> jobs;

    mutex_handle_t queue_mutex;
    cond_handle_t  work_available;
    cond_handle_t  idle_condition;

    bool should_terminate;
    atomic_int_t active_jobs;
} thread_pool_t;

void atomic_inc(atomic_int_t* var)
{
    #ifdef _WIN32
        InterlockedIncrement(var);
    #else
        atomic_fetch_add(var, 1);
    #endif
}

void atomic_dec(atomic_int_t* var)
{
    #ifdef _WIN32
        InterlockedDecrement(var);
    #else
        atomic_fetch_sub(var, 1);
    #endif
}

int atomic_load_int(atomic_int_t* var)
{
    #ifdef _WIN32
        return InterlockedCompareExchange(var, 0, 0);  // Atomic read trick
    #else
        return atomic_load(var);
    #endif
}

thread_handle_t create_thread(thread_func_t func, thread_func_param_t data)
{
    #ifdef _WIN32
        return CreateThread(NULL,  // security attribute -no idea- NULL means default 
                            0,     // stack size zero means default size 
                            func,  // pointer to the function to be executed by the thread
                            data,  // pointer to the params passed to the thread
                            0,     // run immediately
                            NULL); // dont return identifier
    #else
        pthread_t thread;
        pthread_create(&thread, NULL, func, data);
        return thread;
    #endif
}

void join_thread(thread_handle_t thread) 
{
    #ifdef _WIN32
        WaitForSingleObject(thread, INFINITE);  // return only when thread is signaled
        CloseHandle(thread);
    #else
        pthread_join(thread, NULL);
    #endif
}

void mutex_init(mutex_handle_t* mutex)
{
    #ifdef _WIN32
        InitializeCriticalSection(mutex);
    #else
        pthread_mutex_init(mutex, NULL);
    #endif
}

void mutex_destroy(mutex_handle_t* mutex)
{
    #ifdef _WIN32
        DeleteCriticalSection(mutex);
    #else
        pthread_mutex_destroy(mutex);
    #endif
}

void mutex_lock(mutex_handle_t* mutex)
{
    #ifdef _WIN32
        EnterCriticalSection(mutex);
    #else
        pthread_mutex_lock(mutex);
    #endif
}

void mutex_unlock(mutex_handle_t* mutex)
{
    #ifdef _WIN32
        LeaveCriticalSection(mutex);
    #else
        pthread_mutex_unlock(mutex);
    #endif
}

void cond_init(cond_handle_t* cond)
{
    #ifdef _WIN32
        InitializeConditionVariable(cond);
    #else
        pthread_cond_init(cond, NULL);
    #endif
}

void cond_destroy(cond_handle_t* cond)
{
    #ifdef _WIN32
        (void) cond; // no cleanup ?
    #else
        pthread_cond_destroy(cond);
    #endif
}

void cond_wait(cond_handle_t* cond, mutex_handle_t* mutex)
{
    #ifdef _WIN32
        SleepConditionVariableCS(cond, mutex, INFINITE);
    #else
        pthread_cond_wait(cond, mutex);
    #endif
}

void cond_signal(cond_handle_t* cond)
{
    #ifdef _WIN32
        WakeConditionVariable(cond);
    #else
        pthread_cond_signal(cond);
    #endif
}

void cond_broadcast(cond_handle_t* cond)
{
    #ifdef _WIN32
        /* 
            The WakeAllConditionVariable wakes all waiting threads
            while the WakeConditionVariable wakes only a single thread. 
        */
        WakeAllConditionVariable(cond);
    #else
        pthread_cond_broadcast(cond);
    #endif
}

int get_core_count(void) 
{
    #ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return sysinfo.dwNumberOfProcessors;
    #else
        return sysconf(_SC_NPROCESSORS_ONLN);
    #endif
}

thread_func_ret_t thread_loop(thread_func_param_t param)
{
    thread_pool_t* pool = (thread_pool_t*)param;

    job_t job = {0};

    while (true)
    {
        mutex_lock(&pool->queue_mutex);
        {
            /*
                sleep as long as there are no pending jobs
             */
            while (pool->jobs.empty() &&
                   !pool->should_terminate)
            {
                cond_wait(&pool->work_available, &pool->queue_mutex);
            }

            if (pool->should_terminate)
            {
                mutex_unlock(&pool->queue_mutex);
                return 0;
            }
            /*
                pull the job from the queue 
             */
            job = pool->jobs.back();
            pool->jobs.pop_back();
        }
        mutex_unlock(&pool->queue_mutex);

        atomic_inc(&pool->active_jobs);
        job.func(job.data);
        atomic_dec(&pool->active_jobs);

        mutex_lock(&pool->queue_mutex);
        {
            /*
                No jobs left
            */
            if (pool->jobs.empty() &&
                pool->active_jobs == 0)
            {
                cond_broadcast(&pool->idle_condition);
            }
        }
        mutex_unlock(&pool->queue_mutex);
    }

    return 0;
}

thread_pool_t* threadpool_create(void)
{
    thread_pool_t* pool = new thread_pool_t();
    if (!pool) return nullptr;

    pool->should_terminate = false;
    pool->active_jobs = 0;

    mutex_init(&pool->queue_mutex);

    cond_init(&pool->work_available);
    cond_init(&pool->idle_condition);

    uint32_t num_threads = get_core_count();
    pool->threads.reserve(num_threads);

    for (uint32_t i = 0; i < num_threads; ++i) {
        thread_handle_t thread = create_thread(thread_loop, (thread_func_param_t)pool);
        pool->threads.push_back(thread);
    }

    return pool;
}

void threadpool_destroy(thread_pool_t* pool)
{
    if (!pool) return;

    mutex_lock(&pool->queue_mutex);
    {
        pool->should_terminate = true;
    }
    mutex_unlock(&pool->queue_mutex);

    cond_broadcast(&pool->work_available);

    for (size_t i = 0; i < pool->threads.size(); ++i) {
        join_thread(pool->threads[i]);
    }

    mutex_destroy(&pool->queue_mutex);

    cond_destroy(&pool->work_available);
    cond_destroy(&pool->idle_condition);

    delete pool;
    // pool->threads and pool->jobs (std::vector) free themselves automatically
}

void threadpool_queue_job(thread_pool_t* pool, job_func_t func, void* data)
{
    job_t job = { func, data };

    mutex_lock(&pool->queue_mutex);
    {
        pool->jobs.push_back(job);
    }
    mutex_unlock(&pool->queue_mutex);

    // notify a thread to pick up the job
    cond_signal(&pool->work_available);
}

void threadpool_queue_jobs_batch(thread_pool_t* pool, const job_t* jobs, int count)
{
    mutex_lock(&pool->queue_mutex);
    {
        pool->jobs.insert(pool->jobs.end(), jobs, jobs + count);
    }
    mutex_unlock(&pool->queue_mutex);

    cond_broadcast(&pool->work_available); // wake all workers once
}

void threadpool_wait(thread_pool_t* pool)
{
    mutex_lock(&pool->queue_mutex);
    {
        while (!pool->jobs.empty() ||
                pool->active_jobs > 0)
        {
            cond_wait(&pool->idle_condition, &pool->queue_mutex);
        }
    }
    mutex_unlock(&pool->queue_mutex);
}

/*
    there is neither pending nor in progress jobs
 */
bool threadpool_busy(thread_pool_t* pool)
{
    bool busy = false;

    mutex_lock(&pool->queue_mutex);
    {
        busy = (!pool->jobs.empty() ||
                pool->active_jobs > 0);
    }
    mutex_unlock(&pool->queue_mutex);

    return busy;
}

/* ---------------------------------------------------------------- */

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
    ZoneScopedNC("Matrix Multiplication Threaded", tracy::Color::Tomato);
    assert(n_in % 32 == 0);
    u32 blocks_per_row = n_in / 32;

    u32 num_threads = (u32)pool->threads.size();
    u32 rows_per_chunk = (n_out + num_threads - 1) / num_threads;

    // sized once per call; no growth reallocation since capacity == num_threads worst case
    static thread_local std::vector<matmul_chunk_t> chunks;
    static thread_local std::vector<job_t> jobs;
    chunks.clear();
    jobs.clear();
    chunks.reserve(num_threads);
    jobs.reserve(num_threads);

    for (u32 t = 0; t < num_threads; t++)
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


std::string build_chat_prompt(const std::string &system_prompt,
                               const std::string &user_prompt)
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
        u32 head_dim        = model.cfg.embedding_length / model.cfg.attention_head_count;
        u32 kv_dim          = model.cfg.attention_head_count_kv * head_dim;    // 256
        u32 gqa_group_size  = model.cfg.attention_head_count / model.cfg.attention_head_count_kv; // 8
        u32 max_seq         = d_model;
        u32 ffn_dim         = model.cfg.feed_forward_length;
        u32 vocab           = model.token_embd.dims[1];
        f32 scale           = 1.0f / sqrtf((f32)head_dim);
        
        f32 *kv_cache_k     = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
        f32 *kv_cache_v     = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * n_layers * max_seq * kv_dim);
        
        f32 *attn_scores    = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * max_seq);
        f32 *gate           = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
        f32 *up             = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * ffn_dim);
        f32 *logits         = (f32*)ARENA_ALLOC(scratch_arena,sizeof(f32) * vocab);
        
        f32 *q_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *k_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * kv_dim);    
        f32 *v_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * kv_dim);
        f32 *o_full         = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        f32 *normed_attn    = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *attn_out       = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        
        f32 *residual       = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *normed_ffn     = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *ffn_out        = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
        f32 *normed_final   = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    

        f32 *x              = (f32*)ARENA_ALLOC(scratch_arena, sizeof(f32) * d_model);    
    parse.Leave();
        
    TracySection promp_tokenize("Promp Tokenization");
        init_tokenizer(); 
        std::string prompt = build_chat_prompt(
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            "Give me a python script to plot a sine function."
        );
        std::vector<int> tokens =  encode(prompt);
        std::println("{}",tokens);

        for(int token: tokens){
            std::println("{}",decode_id(token));    
        }    

        u32 prompt_len = (u32)tokens.size();
        u32 max_new_tokens = 200;
    promp_tokenize.Leave();
    

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
                
                mat_vec_mul_q8_0((block_q8_0*)l.attn_q.tensor_data, l.attn_q.dims[0], l.attn_q.dims[1], normed_attn, q_full);
                for (u32 i = 0; i < d_model; i++) {
                    q_full[i] += ((f32*)l.attn_q_bias.tensor_data)[i];
                }

                mat_vec_mul_q8_0((block_q8_0*)l.attn_k.tensor_data, l.attn_k.dims[0], l.attn_k.dims[1], normed_attn, k_full);
                for (u32 i = 0; i < 256; i++){
                    k_full[i] += ((f32*)l.attn_k_bias.tensor_data)[i];
                }

                mat_vec_mul_q8_0((block_q8_0*)l.attn_v.tensor_data, l.attn_v.dims[0], l.attn_v.dims[1],  normed_attn, v_full);
                for (u32 i = 0; i < 256; i++){
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
                        for (u32 i = 0; i < head_dim; i++) oh[i] = 0.0f;
                        for (u32 t = 0; t <= cur_pos; t++)
                        {
                            f32 *vh_t = kv_cache_v + (layer_idx * max_seq + t) * kv_dim + kv*head_dim;
                            f32 w = attn_scores[t];
                            for (u32 i = 0; i < head_dim; i++)
                            {
                                oh[i] += w * vh_t[i];
                            }
                        }
                    }
                }
                
                mat_vec_mul_q8_0((block_q8_0*)l.attn_output.tensor_data, l.attn_output.dims[0], l.attn_output.dims[1], o_full, attn_out);

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
                if (next_token == 151645 || next_token == 151643)
                    break;
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