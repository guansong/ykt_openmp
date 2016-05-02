#include "../src/Intrinsic.cl"

#include "../src/cancel.cl"
#include "../src/critical.cl"
#include "../src/debug.cl"
#include "../src/libcall.cl"
#include "../src/loop.cl"
#include "../src/omptarget-hsail.cl"
#include "../src/parallel.cl"
#include "../src/stdio.cl"
#include "../src/sync.cl"
#include "../src/task.cl"

// Can we have global at file scope?
__global int threadsForPar = 0;

// Can we have local at file scope?
//__local int counter;

// Can we have global atomic
__global atomic_int lock;

// Can we have local atomic
//__local atomic_int lock;

__kernel void copy_helloworld(__global char* in, __global char* out)
{
#if 0
    int num = get_local_id(0);

    if  (num<5) {
        out[num] = in[num]+1;
        PRINT(LD_IO,"%s from thread %d\n", "used", num);
    } else {
        PRINT(LD_IO,"%s from thread %d\n", "restricted", num);
    }
#endif

    int num = __kmpc_ocl_get_local_id();

    out[num] = in[num] + 1;
}

// Can we have a lock
void test_lock(__global char* in, __global char* out)
{
    __private kmp_Indent * loc;

    // Use OpenCL way of load store atomic value
    // Use OpenMP way of unlock
    if (!__kmpc_ocl_get_global_id()) {

        atomic_store(&lock, 1);

        omp_unset_lock(&lock);

        PRINT(LD_IO,"%s from thread %d: %d\n", "test lock" , __kmpc_ocl_get_local_id(), atomic_load(&lock));
    }

    // more test to be added
}

// Can we malloc
void test_malloc(__global char* in, __global char* out)
{
    int num = __kmpc_ocl_get_local_id();
    //if (!num) {
    //    __global char * ptr = (__global char *)malloc(10);
    //}

    // do we need this?
    //if (!num) {
    //    __local char * ptr = (__local char *)malloc(10);
    //}
}

// Can we init the runtime
void test_init(__global char* in, __global char* out)
{
    __private kmp_Indent * loc;

    //init lib
    if (!__kmpc_ocl_get_global_id()) {

        //init runtime
        // is this globa size or local size?
        __kmpc_kernel_init(0, get_global_size(0));
    }
}

// Can we have a parallel with restricted number of threads
void test_parallel(__global char* in, __global char* out)
{
    __private kmp_Indent * loc;

    int num = __kmpc_ocl_get_local_id();

    if (!__kmpc_ocl_get_local_id()) {
        //set thread limit
        //loc, not used
        //0, id
        //thread number
        __kmpc_push_num_threads(loc, 0, 5);

        //prepare 0, num not used
        //simdlane
        threadsForPar = __kmpc_kernel_prepare_parallel(0, 1);

        PRINT(LD_IO,"Participating number of threads %d\n", threadsForPar);
    }

    if (num < threadsForPar)
    {
        PRINT(LD_IO,"%s from thread %d\n", "used", num);

        //parallel
        //simdlane
        __kmpc_kernel_parallel(1);

        // original computation
        copy_helloworld(in, out);

        //end parallel
        __kmpc_kernel_end_parallel();
    }
    else
    {
        PRINT(LD_IO,"%s from thread %d\n", "restricted", num);

        // do nothing for these threads;

        // for test only
        // out[num] = 0;
        // we should set the value on the host actually
        // before print, the host should ensure the value is 0, so nothing extra is print out
    }
}

__kernel void helloworld(__global char* in, __global char* out)
{
    __private kmp_Indent * loc;

    // Can we show id
    PRINT(LD_IO,"%s from thread %d\n", "test" , __kmpc_ocl_get_local_id());

    // Can we init the runtime
    test_init(in, out);

    // Can we do barrier
    // loc
    // id, not used
    __kmpc_barrier (loc, __kmpc_ocl_get_global_id());

    // Can we have a lock
    test_lock(in, out);

    //Can we malloc?
    test_malloc(in, out);

    // Can we do parallel with the right number of threads;
    test_parallel(in, out);

    // More test on par do
}

