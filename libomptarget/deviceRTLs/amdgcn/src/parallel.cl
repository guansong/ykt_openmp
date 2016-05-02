//===--- parallel.cl - HSAIL OpenMP GPU initialization ---- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//
//
// Parallel implemention in the GPU. Here is the pattern:
//
//    while (not finished) {
//
//    if (master) {
//      sequential code, decide which par loop to do, or if finished
//     __kmpc_kernel_prepare_parallel() // exec by master only
//    }
//    syncthreads // A
//    __kmpc_kernel_parallel() // exec by all
//    if (this thread is included in the parallel) {
//      switch () for all parallel loops
//      __kmpc_kernel_end_parallel() // exec only by threads in parallel
//    }
//
//
//    The reason we don't exec end_parallel for the threads not included
//    in the parallel loop is that for each barrier in the parallel
//    region, these non-included threads will cycle through the
//    syncthread A. Thus they must preserve their current threadId that
//    is larger than thread in team.
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"

////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes parallel (1 static level only)
////////////////////////////////////////////////////////////////////////////////

// return number of hsa threads that participate to parallel
// calculation has to consider simd implementation in hsail
// i.e. (num omp threads * num lanes)
//
// hsathreads =
//    if(num_threads != 0) {
//      if(thread_limit > 0) {
//        min (num_threads*numLanes ; thread_limit*numLanes);
//      } else {
//        min (num_threads*numLanes; blockDim.x)
//      }
//    } else {
//      if (thread_limit != 0) {
//        min (thread_limit*numLanes; blockDim.x)
//      } else { // no thread_limit, no num_threads, use all cuda threads
//        blockDim.x;
//      }
//    }
EXTERN int __kmpc_kernel_prepare_parallel(int NumThreads, int NumLanes)
{
    PRINT0(LD_IO , "call to __kmpc_kernel_prepare_parallel\n");

    int globalThreadId = GetGlobalThreadId();
    int globalTeamId = GetGlobalTeamId();

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr =
        omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId];

    ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");

    if (currTaskDescr->data.items.flags & TaskDescr_InPar) {
        PRINT0(LD_PAR, "already in parallel: go seq\n");

        // todo: support nested parallelism
        // if we have nested parallel region, the env may have issues
        // as the topLevelTaskDescriptor does not have a new one?
        return 0;
    }

    uint16_t HSAThreadsForParallel = 0;
    uint16_t NumThreadsClause =
        omptarget_hsail_threadPrivateContext.nthreadsForNextPar[globalThreadId];

    PRINT(LD_IO , "NumThreadsClause %d\n", NumThreadsClause);

    // we cannot have more than block size
    uint16_t HSAThreadsAvail = GetNumberOfThreadsInBlock();
    PRINT(LD_IO , "Threads available %d\n", HSAThreadsAvail);

    // this is different from ThreadAvail of OpenMP because we may be
    // using some of the HSA threads as SIMD lanes

    if (NumThreadsClause != 0) {
        // reset request to avoid propagating to successive #parallel
        omptarget_hsail_threadPrivateContext.nthreadsForNextPar[globalThreadId]=0;

        // assume that thread_limit*numlanes is already <= HSAThreadsAvail
        // because that is already checked on the host side (HSA offloading rtl)
        if (currTaskDescr->data.items.threadlimit != 0)
            HSAThreadsForParallel =
                NumThreadsClause*NumLanes < currTaskDescr->data.items.threadlimit*NumLanes ?
                NumThreadsClause*NumLanes : currTaskDescr->data.items.threadlimit*NumLanes;
        else {
            HSAThreadsForParallel = (NumThreadsClause*NumLanes > HSAThreadsAvail) ?
                HSAThreadsAvail : NumThreadsClause*NumLanes;
        }

        PRINT(LD_IO , "Threads for parallel %d\n", HSAThreadsForParallel);
    } else {
        if (currTaskDescr->data.items.threadlimit != 0) {
            HSAThreadsForParallel =
                (currTaskDescr->data.items.threadlimit*NumLanes > HSAThreadsAvail) ?
                HSAThreadsAvail : currTaskDescr->data.items.threadlimit*NumLanes;
        } else {
            HSAThreadsForParallel = GetNumberOfThreadsInBlock();
        }

        PRINT(LD_IO , "Threads for parallel %d\n", HSAThreadsForParallel);
    }

    ASSERT(LT_FUSSY, HSAThreadsForParallel > 0, "bad thread request of %d threads", HSAThreadsForParallel);
    ASSERT0(LT_FUSSY, GetThreadIdInBlock() == TEAM_MASTER, "only team master can create parallel");

    // set number of threads on work descriptor
    // copy the current task descr to the one in WorkDescr
    // this is different from the number of hsail threads required for the parallel region
    __MSPACE omptarget_hsail_WorkDescr * workDescr = getMyWorkDescriptor();
    TaskDescr_CopyToWorkDescr(&workDescr->masterTaskICV, currTaskDescr, HSAThreadsForParallel/NumLanes);

#if 0
    // init counters (copy start to init)
    workDescr.CounterGroup().Reset();
#endif

    PRINTTASKDESCR("Team master", currTaskDescr);

    return HSAThreadsForParallel;
}

// works only for active parallel loop...
EXTERN void __kmpc_kernel_parallel(int numLanes)
{
    PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_parallel\n");

    // init work descriptor from workdesccr
    int globalThreadId = GetGlobalThreadId();
    int globalTeamId = GetGlobalTeamId();

    __MSPACE omptarget_hsail_WorkDescr * workDescr = getMyWorkDescriptor();

    __MSPACE omptarget_hsail_TaskDescr * workMaster = &workDescr->masterTaskICV;

    PRINTTASKDESCR("Work master", workMaster);

    __MSPACE omptarget_hsail_TaskDescr *newTaskDescr =
        &omptarget_hsail_threadPrivateContext.levelOneTaskDescr[globalThreadId];

    ASSERT0(LT_FUSSY, newTaskDescr, "expected a task descr");

    TaskDescr_CopyFromWorkDescr(newTaskDescr, &workDescr->masterTaskICV);

    // install new top descriptor
    omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId]=newTaskDescr;

#if 0
    // init private from int value
    workDescr.CounterGroup().Init(omptarget_hsail_threadPrivateContext.Priv(globalThreadId));
#endif

    PRINT(LD_PAR, "thread will execute parallel region with id %d in a team of %d threads\n",
            newTaskDescr->data.items.threadId, newTaskDescr->data.items.nthreads);

    // each thread sets its omp thread ID when entering a parallel region
    // based on the number of simd lanes and its cuda thread ID
    if (numLanes > 1) {
        // the compiler is requesting lanes for #simd execution
        // WARNING: assume thread number in #parallel is a multiple of numLanes
        newTaskDescr->data.items.threadId /= numLanes;
        //newTaskDescr->ThreadsInTeam(); // =  newTaskDescr->ThreadsInTeam()/numLanes;
    }
    //  } else {
    //    // not a for with a simd inside: use only one lane
    //    // we may have started thread_limit*simd_info HSA threads
    //    // and we need to set the number of threads to thread_limit value
    //    // FIXME: is this always the case, even if numLanes > 1?
    ////    newTaskDescr->ThreadId() = threadIdx.x;
    //    //newTaskDescr->ThreadsInTeam();// = newTaskDescr->ThreadLimit();
    //  }

    PRINTTASKDESCR("Top level", newTaskDescr);
}

EXTERN void __kmpc_kernel_end_parallel()
{
    PRINT0(LD_IO | LD_PAR, "call to __kmpc_kernel_end_parallel\n");

    int globalThreadId = GetGlobalThreadId();
    int globalTeamId = GetGlobalTeamId();

    // pop stack
    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptorById(globalThreadId);

    omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId]=currTaskDescr->prev;
}


////////////////////////////////////////////////////////////////////////////////
// support for parallel that goes sequential
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_serialized_parallel (
        __private kmp_Indent *loc,
        uint32_t global_tid)
{
    PRINT0(LD_IO, "call to __kmpc_serialized_parallel\n");

#if 0
    // assume this is only called for nested parallel
    int globalThreadId = GetGlobalThreadId();
    int globalTeamId = GetGlobalTeamId();

    // unlike actual parallel, threads in the same team do not share
    // the workTaskDescr in this case and num threads is fixed to 1

    // get current task
    omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(globalThreadId);

    // allocate new task descriptor and copy value from current one, set prev to it
    omptarget_hsail_TaskDescr *newTaskDescr = (omptarget_hsail_TaskDescr *)
        SafeMalloc(sizeof(omptarget_hsail_TaskDescr), (char *) "new seq parallel task");
    newTaskDescr->CopyParent(currTaskDescr);

    // tweak values for serialized parallel case:
    // - each thread becomes ID 0 in its serialized parallel, and
    // - there is only one thread per team
    newTaskDescr->ThreadId() = 0;
    newTaskDescr->ThreadsInTeam() = 1;

    // set new task descriptor as top
    omptarget_hsail_threadPrivateContext.SetTopLevelTaskDescr(globalThreadId, newTaskDescr);
#endif
}

EXTERN void __kmpc_end_serialized_parallel (
        __private kmp_Indent *loc,
        uint32_t global_tid)
{
    PRINT0(LD_IO, "call to __kmpc_end_serialized_parallel\n");

#if 0
    // pop stack
    int globalThreadId = GetGlobalThreadId();
    omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor(globalThreadId);
    // set new top
    omptarget_hsail_threadPrivateContext.SetTopLevelTaskDescr(globalThreadId,
            currTaskDescr->GetPrevTaskDescr());
    // free
    SafeFree(currTaskDescr, (char *) "new seq parallel task");
#endif
}

////////////////////////////////////////////////////////////////////////////////
// push params
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_push_num_threads (
        __private kmp_Indent * loc,
        int gtid,
        int num_threads)
{
    PRINT(LD_IO, "call kmpc_push_num_threads %d\n", num_threads);

    int globalThreadId = GetGlobalThreadId();
    omptarget_hsail_threadPrivateContext.nthreadsForNextPar[globalThreadId] = num_threads;
}

// Do not do nothing: the host guarantees we started the requested number of
// teams and we only need inspection gridDim

EXTERN void __kmpc_push_num_teams (
        __private kmp_Indent * loc,
        int gtid,
        int num_teams,
        int thread_limit)
{
    PRINT(LD_IO, "call kmpc_push_num_teams %d\n", num_teams);
    ASSERT0(LT_FUSSY, FALSE, "should never have anything with new teams on device");
}

