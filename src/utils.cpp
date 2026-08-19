#include "utils.hpp"

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

int atomic_exchange(atomic_int_t* var, int value)
{
#ifdef _WIN32
    return InterlockedExchange(var, value);
#else
    return atomic_exchange(var, value);
#endif
}

int atomic_cas(atomic_int_t* var, int expected, int desired)
{
#ifdef _WIN32
    return InterlockedCompareExchange(var, desired, expected) == expected;
#else
    int e = expected;
    return atomic_compare_exchange_strong(var, &e, desired);
#endif
}

void cpu_relax(void)
{
#ifdef _WIN32
    YieldProcessor(); YieldProcessor(); YieldProcessor(); YieldProcessor();
#elif defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    /* fall back to compiler barrier */
    __asm__ __volatile__("" ::: "memory");
#endif
}

void os_yield(void)
{
#ifdef _WIN32
    SwitchToThread();        /* yield to a thread on the same CPU */
#else
    sched_yield();
#endif
}
void spin_init(spinlock_t* s)    
{ 
    atomic_exchange(&s->locked, 0); 
}

void spin_lock(spinlock_t* s)
{
    for (;;) {
        if (!atomic_exchange(&s->locked, 1))
            return;
        while (atomic_load_int(&s->locked))
            cpu_relax();
    }
}

void spin_unlock(spinlock_t* s)
{
    atomic_exchange(&s->locked, 0);
}
thread_func_ret_t thread_loop(thread_func_param_t param)
{
    thread_pool_t* pool = (thread_pool_t*)param;

    job_t job = {};

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

    // uint32_t num_threads = get_core_count();
    uint32_t num_threads = 6;
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

