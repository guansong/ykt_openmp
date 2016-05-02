//===--- omptarget-hsail.cl - HSAIL OpenMP GPU initialization ---- HSA -*-===//
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

#include "omptarget-hsail.h"

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

__MSPACE omptarget_hsail_TeamPrivateContext omptarget_hsail_teamPrivateContext;
__MSPACE omptarget_hsail_ThreadPrivateContext omptarget_hsail_threadPrivateContext;
__MSPACE omptarget_hsail_GlobalICV omptarget_hsail_globalICV;

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_kernel_init(int OmpHandle, int ThreadLimit)
{
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f\n", OMPTARGET_HSAIL_VERSION);
  // init thread private
  int globalThreadId = GetGlobalThreadId();
  ThreadPrivateContext_InitThreadPrivateContext(&omptarget_hsail_threadPrivateContext,globalThreadId);

  int threadIdInBlock = GetThreadIdInBlock();
  if (threadIdInBlock == TEAM_MASTER) {
    PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");
    PRINT(LD_IO, "size of TeamDesc %ldx%d\n", sizeof(omptarget_hsail_TeamDescr), MAX_NUM_TEAMS);
    PRINT(LD_IO, "size of TaskDesc %ldx%d\n", sizeof(omptarget_hsail_TaskDescr), MAX_NUM_OMP_THREADS);

    //PRINT(LD_IO, "size of teamcontext %ld\n", sizeof(omptarget_hsail_teamPrivateContext));
    //PRINT(LD_IO, "size of threadcontext %ld\n", sizeof(omptarget_hsail_threadPrivateContext));
    //PRINT(LD_IO, "size of globalICV %ld\n", sizeof(omptarget_hsail_globalICV));

    //PRINT(LD_IO, "size of kmp_CriticalName %ld\n", sizeof(kmp_CriticalName));

    // init global icv
    omptarget_hsail_globalICV.gpuCycleTime = 1.0 / 745000000.0; // host reports 745 mHz
    omptarget_hsail_globalICV.cancelPolicy = FALSE;  // currently false only
    // init team context
    __MSPACE omptarget_hsail_TeamDescr * currTeamDescr = getMyTeamDescriptor();
    TeamDescr_InitTeamDescr(currTeamDescr);

    // this thread will start execution... has to update its task ICV
    // to points to the level zero task ICV. That ICV was init in
    omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId]=
        &currTeamDescr->levelZeroTaskDescr;

    // set number of threads and thread limit in team to started value
    int globalThreadId = GetGlobalThreadId();
    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr =
        omptarget_hsail_threadPrivateContext.topLevelTaskDescrPtr[globalThreadId];

    currTaskDescr->data.items.nthreads = GetNumberOfThreadsInBlock();

    currTaskDescr->data.items.threadlimit = ThreadLimit;

    PRINTTASKDESCR("Initial", currTaskDescr);
  }
}

