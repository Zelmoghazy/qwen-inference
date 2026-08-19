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
