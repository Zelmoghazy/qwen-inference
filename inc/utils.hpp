#pragma once

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN 
    #define VC_EXTRALEAN
    #define NOMINMAX
    #include <windows.h>
#endif

#include <cstdint>
#include <cassert>
#include <print>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <immintrin.h> 

#include "tracy/tracy/Tracy.hpp"

#define LOG 1

typedef uint8_t u8;
typedef int8_t i8;
typedef uint16_t u16;
typedef uint64_t u64;
typedef int64_t i64;
typedef uint32_t u32;
typedef float f32;
typedef double f64;


typedef __m256  f32x8;

#define ALIGN_UP(x, y)                  ((((x) + (y) - 1) / (y)) * (y))

struct TracySection
{
    explicit TracySection( const char* name ) { Enter( name ); }
    explicit TracySection( const char* name, uint16_t category ) : idx( TracySectionEnterCategory( category, "%s", name ) ) {}
    ~TracySection() { Leave(); }

    void Enter( const char* name )
    {
        idx = TracySectionEnter( "%s", name );
    }

    void Leave()
    {
        if( idx > 0 )
        {
            TracySectionLeave( idx );
            idx = 0;
        }
    }

private:
    uint32_t idx;
};


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

typedef struct {
    atomic_int_t locked;
} spinlock_t;

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

void atomic_inc(atomic_int_t* var);
void atomic_dec(atomic_int_t* var);
int atomic_load_int(atomic_int_t* var);
thread_handle_t create_thread(thread_func_t func, thread_func_param_t data);
void join_thread(thread_handle_t thread);
void mutex_init(mutex_handle_t* mutex);
void mutex_destroy(mutex_handle_t* mutex);
void mutex_lock(mutex_handle_t* mutex);
void mutex_unlock(mutex_handle_t* mutex);
void cond_init(cond_handle_t* cond);
void cond_destroy(cond_handle_t* cond);
void cond_wait(cond_handle_t* cond, mutex_handle_t* mutex);
void cond_signal(cond_handle_t* cond);
void cond_broadcast(cond_handle_t* cond);
int get_core_count(void);
thread_func_ret_t thread_loop(thread_func_param_t param);
thread_pool_t* threadpool_create(void);
void threadpool_destroy(thread_pool_t* pool);
void threadpool_queue_job(thread_pool_t* pool, job_func_t func, void* data);
void threadpool_queue_jobs_batch(thread_pool_t* pool, const job_t* jobs, int count);
void threadpool_wait(thread_pool_t* pool);

/*
    there is neither pending nor in progress jobs
 */
bool threadpool_busy(thread_pool_t* pool);



static inline f32x8 f32x8_zero(void) {
    return _mm256_setzero_ps();
}
static inline f32x8 f32x8_set1(float x) 
{
    return _mm256_set1_ps(x);
}

static inline f32x8 f32x8_load_i8(const int8_t* ptr) 
{
    __m128i i8 = _mm_loadl_epi64((const __m128i*)ptr);
    __m256i i32 = _mm256_cvtepi8_epi32(i8);
    return _mm256_cvtepi32_ps(i32);
}

static inline f32x8 f32x8_load(const float *ptr) 
{
    return _mm256_loadu_ps(ptr);
}

static inline f32x8 f32x8_madd(f32x8 a, f32x8 b, f32x8 c) 
{
    // a * b + c
    return _mm256_fmadd_ps(a, b, c);
}

static inline float f32x8_sum(f32x8 v) 
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
