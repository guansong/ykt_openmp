//===------------ debug.cu - hsail OpenMP debug utilities --------HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of debug utilities to be
// used in the application.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"

////////////////////////////////////////////////////////////////////////////////
// print current state
////////////////////////////////////////////////////////////////////////////////

NOINLINE void PrintTaskDescr(
        __MSPACE omptarget_hsail_TaskDescr *taskDescr,
        __constant char * title,
        int level)
{

#if OMPTARGET_HSAIL_DEBUG
    //omp_sched_t sched = taskDescr->GetRuntimeSched();
    omp_sched_t sched = (taskDescr->data.items.flags & TaskDescr_SchedMask) + 1;

    if (DON(LD_ALL))
        printf("Task %p(%d): ispar %x, inpar %x, dyn %d, sched %d, nthds %d, tlt %d, tid %d, team %d, chunk %ld (%p)\n",
            taskDescr,
            level,
            (taskDescr->data.items.flags & TaskDescr_IsParConstr)/TaskDescr_IsParConstr,
            (taskDescr->data.items.flags & TaskDescr_InPar)/TaskDescr_InPar,
            (taskDescr->data.items.flags & TaskDescr_IsDynamic)/TaskDescr_IsDynamic,
            sched,
            (int)taskDescr->data.items.nthreads,
            (int)taskDescr->data.items.threadlimit,
            (int)taskDescr->data.items.threadId,
            (int)taskDescr->data.items.threadsInTeam,
            taskDescr->data.items.runtimeChunkSize,
            taskDescr->prev);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// debug for compiler (should eventually all vanish)
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_print_str(char *title)
{
  PRINT(LD_ALL, " %s\n", title);
}

EXTERN void __kmpc_print_title_int(char *title, int data)
{
  PRINT(LD_ALL, "%s val=%d\n", title, data);
}

EXTERN void __kmpc_print_index(char *title, int i)
{
  PRINT(LD_ALL, "i = %d\n", i);
}

EXTERN void __kmpc_print_int(int data)
{
  PRINT(LD_ALL, "val=%d\n", data);
}

EXTERN void __kmpc_print_double(double data)
{
  PRINT(LD_ALL, "val=%lf\n", data);
}

EXTERN void __kmpc_print_address_int64(int64_t data)
{
  PRINT(LD_ALL, "val=%016llx\n", data);
}

////////////////////////////////////////////////////////////////////////////////
// substitute for printf in kernel (should vanish)
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_kernel_print(char *title)
{
  PRINT(LD_ALL, " %s\n", title);
}

EXTERN void __kmpc_kernel_print_int8(char *title, int64_t data)
{
  PRINT(LD_ALL, " %s val=%lld\n", title, data);
}

