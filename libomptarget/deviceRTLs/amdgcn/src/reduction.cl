//===---- reduction.cl - HSAIL OpenMP reduction implementation ---- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of reduction with KMPC interface.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"

//cannot implement atomic_start and atomic_end for GPU. Report runtime error
EXTERN void __kmpc_atomic_start() {
    ASSERT0(LT_FUSSY, 0, "__kmpc_atomic_start not supported\n");
}

EXTERN void __kmpc_atomic_end() {
    ASSERT0(LT_FUSSY, 0, "__kmpc_atomic_end not supported\n");
}

EXTERN
int32_t __gpu_block_reduce(){
#if 0
    if (omp_get_num_threads() != blockDim.x)
        return 0;
    unsigned tnum = __ballot(1);
    if (tnum != (~0x0)) { //assume swapSize is 32
        return 0;
    }

    return 1;
#endif

    return 0;
}

EXTERN
int32_t __kmpc_reduce_gpu (
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t num_vars,
        size_t reduce_size,
        void *reduce_data,
        kmp_ReductFctPtr *reductFct,
        __local kmp_CriticalName *lck) {

    int globalThreadId = GetGlobalThreadId();

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr =
        omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId];

    int numthread;

    if (currTaskDescr->data.items.flags & TaskDescr_InPar) {
        numthread = omp_get_num_threads();
    } else {
        numthread = omp_get_num_teams();
    }

    if (numthread == 1) {
        return 1;
    }
    else if (!__gpu_block_reduce()) {
        return 2;
    }
    else {
        if (get_local_id(0) == 0)
            return 1;
        else
            return 0;
    }

    //FIXME: ???
    //	return 2;
    /**
     * Only when all the threads in a block are doing reduction,
     * the warpBlockRedu is used. Otherwise atomic.
     * check the data type, too.
     * A special case: when the size of thread group is one,
     * do reduction directly.
     **/

    // Note: this code provokes warning because it follows a "return"

    //since there is no thread interface yet, just infer from the
    // result of ballot
#if 0
    unsigned tnum = __ballot(1);
    if (tnum != (~0x0)) { //assume swapSize is 32
        return 2;
    }

#if 0
    if (threadIdx.x == 0) {
        if ((void *)reductFct != (void *)omp_reduction_op_gpu) {
            printf("function pointer value is not correct\n");
        } else {
            printf("function pointer value is correct\n");
        }
    }
#endif

    //printf("function pointer %p %d %p\n", reductFct, reduce_size, omp_reduction_op_gpu);
    if (reduce_size == 0) {
        (*reductFct)((char*)reduce_data, (char*)reduce_data);
    } else {
        //omp_reduction_op_gpu((char*)reduce_data, (char*)reduce_data);
        (*gpu_callback)((char*)reduce_data, (char*)reduce_data);
    }

    //int **myp = (int **) reduce_data;
    // the results are with thread 0. Reduce to the shared one
    if (threadIdx.x == 0) {
        //printf("function pointer %p %p\n", reductFct, omp_reduction_op);
        //   	printf("my result %d\n", *myp[0]);
        return 1;
    } else {
        return 0;
    }
#endif

}

EXTERN
int32_t __kmpc_reduce (
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t num_vars,
        size_t reduce_size,
        __private void *reduce_data, //FIXME
        __private kmp_ReductFctPtr *reductFct,
        __local kmp_CriticalName *lck) {

    return __kmpc_reduce_gpu(loc, global_tid, num_vars, reduce_size, reduce_data, reductFct, lck);
}

EXTERN
void __kmpc_end_reduce(
        __private kmp_Indent *loc,
        int32_t global_tid,
        __local kmp_CriticalName *lck ) {
}

EXTERN
int32_t __kmpc_reduce_nowait(
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t num_vars,
        size_t reduce_size,
        __private void *reduce_data, //FIXME
        __private kmp_ReductFctPtr *reductFct,
        __local kmp_CriticalName *lck) {

    int globalThreadId = GetGlobalThreadId();

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr =
        omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId];

    int numthread;

    if (currTaskDescr->data.items.flags & TaskDescr_InPar) {
        numthread = omp_get_num_threads();
    } else {
        numthread = omp_get_num_teams();
    }

    if (numthread == 1) {
        return 1;
    }
    else if (!__gpu_block_reduce()) {
        return 2;
    }
    else {
        if (get_local_id(0) == 0)
            return 1;
        else
            return 0;
    }

    //FIXME: ???
    // Notice: as above, uncomment if 0 once this code below is ready for shipping
#if 0
    unsigned tnum = __ballot(1);
    if (tnum != (~0x0)) { //assume swapSize is 32
        return 2;
    }

    if (threadIdx.x == 0) {
        printf("choose block reduction\n");
    }

    (*reductFct)(reduce_data, reduce_data);
    //omp_reduction_op((char*)reduce_data, (char*)reduce_data);

    int **myp = (int **) reduce_data;
    // the results are with thread 0. Reduce to the shared one
    if (threadIdx.x == 0) {
        printf("my result %d\n", *myp[0]);
        return 1;
    } else {
        return 0;
    }
#endif
}

EXTERN
void __kmpc_end_reduce_nowait(
        __private kmp_Indent *loc,
        int32_t global_tid,
        __local kmp_CriticalName *lck ) {
}


#if 1
// keep for debugging
EXTERN
void __kmpc_atomic_fixed4_add(
        __private kmp_Indent *id_ref,
        int32_t gtid,
        __global int32_t * lhs,
        int32_t rhs) {
    PRINT(LD_LOOP, "thread %d participating in reduction, lhs = %p, rhs = %d\n", gtid, lhs, rhs);
    //atomicAdd(lhs, rhs);
}
#endif



