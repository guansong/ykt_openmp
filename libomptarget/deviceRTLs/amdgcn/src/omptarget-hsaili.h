//===--- omptarget-hsaili.h - HSAIL OpenMP GPU initialization ---- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU for HSAIL
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of all library macros, types,
// and functions.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// Task Descriptor
////////////////////////////////////////////////////////////////////////////////
INLINE omp_sched_t TaskDescr_GetRuntimeSched(
    __MSPACE omptarget_hsail_TaskDescr * desc)
{
  // sched starts from 1..4; encode it as 0..3; so add 1 here
  uint8_t rc = (desc->data.items.flags & TaskDescr_SchedMask) +1;
  return (omp_sched_t) rc;
}

INLINE void TaskDescr_SetRuntimeSched(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    omp_sched_t sched)
{
  // sched starts from 1..4; encode it as 0..3; so add 1 here
  uint8_t val = ((uint8_t) sched) -1;
  // clear current sched
  desc->data.items.flags &= ~TaskDescr_SchedMask;
  // set new sched
  desc->data.items.flags |= val;
}

INLINE void TaskDescr_InitLevelZeroTaskDescr(
    __MSPACE omptarget_hsail_TaskDescr * desc)
{
  // xxx slow method
  /* flag:
      default sched is static,
      dyn is off (unused now anyway, but may need to sample from host ?)
      not in parallel
  */
  desc->data.items.flags = 0;
  desc->data.items.nthreads = GetNumberOfProcsInTeam();; // threads: whatever was alloc by kernel
  desc->data.items.threadId = 0; // is master
  desc->data.items.threadsInTeam = 1; // sequential
  desc->data.items.runtimeChunkSize = 1; // prefered chunking statik with chunk 1
}

INLINE void TaskDescr_CopyData(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    __MSPACE omptarget_hsail_TaskDescr * sourceTaskDescr)
{
  desc->data.vect[0] = sourceTaskDescr->data.vect[0];
  desc->data.vect[1] = sourceTaskDescr->data.vect[1];
  desc->data.vect[2] = sourceTaskDescr->data.vect[2];
}

INLINE void TaskDescr_Copy(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    __MSPACE omptarget_hsail_TaskDescr * sourceTaskDescr)
{
  TaskDescr_CopyData(desc, sourceTaskDescr);
  desc->prev = sourceTaskDescr->prev;
}

INLINE void TaskDescr_CopyParent(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    __MSPACE omptarget_hsail_TaskDescr * parentTaskDescr)
{
  TaskDescr_CopyData(desc, parentTaskDescr);
  desc->prev = parentTaskDescr;
}

INLINE void TaskDescr_CopyForExplicitTask(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    __MSPACE omptarget_hsail_TaskDescr * parentTaskDescr)
{
  TaskDescr_CopyParent(desc, parentTaskDescr);
  desc->data.items.flags &= ~TaskDescr_IsParConstr;
  ASSERT0(LT_FUSSY, desc->data.items.flags & TaskDescr_IsParConstr, "expected task");
}

INLINE void TaskDescr_CopyToWorkDescr(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    __MSPACE omptarget_hsail_TaskDescr * masterTaskDescr,
    uint16_t tnum)
{
  TaskDescr_CopyParent(desc, masterTaskDescr);
  // overrwrite specific items;
  desc->data.items.flags |= TaskDescr_InPar | TaskDescr_IsParConstr; // set flag to parallel
  desc->data.items.threadsInTeam = tnum; // set number of threads
}

INLINE void TaskDescr_CopyFromWorkDescr(
    __MSPACE omptarget_hsail_TaskDescr * desc,
    __MSPACE omptarget_hsail_TaskDescr * workTaskDescr)
{
  TaskDescr_Copy(desc, workTaskDescr);
  // overrwrite specific items;
  desc->data.items.threadId = GetThreadIdInBlock(); // get ids (only called for 1st level)
}


////////////////////////////////////////////////////////////////////////////////
// Thread Private Context
////////////////////////////////////////////////////////////////////////////////
INLINE __MSPACE omptarget_hsail_TaskDescr *ThreadPrivateContext_GetTopLevelTaskDescr(
    __MSPACE omptarget_hsail_ThreadPrivateContext * desc,
    int gtid)
{
  ASSERT0(LT_FUSSY, gtid < MAX_NUM_OMP_THREADS,
      "Getting top level, gtid is larger than allocated data structure size");
  return desc->topLevelTaskDescrPtr[gtid];
}

INLINE void ThreadPrivateContext_SetTopLevelTaskDescr(
    __MSPACE omptarget_hsail_ThreadPrivateContext * desc,
    int gtid,
    __MSPACE omptarget_hsail_TaskDescr * taskDescr)
{
  ASSERT0(LT_FUSSY, gtid < MAX_NUM_OMP_THREADS,
      "Getting top level, gtid is larger than allocated data structure size");
  desc->topLevelTaskDescrPtr[gtid]=taskDescr;
}

INLINE void ThreadPrivateContext_InitThreadPrivateContext(
    __MSPACE omptarget_hsail_ThreadPrivateContext * desc,
    int gtid)
{
  // levelOneTaskDescr is init when starting the parallel region
  // top task descr is NULL (team master version will be fixed separately)
  desc->topLevelTaskDescrPtr[gtid] = 0;
  // no num threads value has been pushed
  desc->nthreadsForNextPar[gtid] = 0;
  // priv counter init to zero
  desc->priv[gtid] = 0;
  // the following don't need to be init here; they are init when using dyn sched
  // current_Event, events_Number, chunk, num_Iterations, schedule
}


////////////////////////////////////////////////////////////////////////////////
// Work Descriptor
////////////////////////////////////////////////////////////////////////////////

INLINE void WorkDescr_InitWorkDescr(
    __MSPACE omptarget_hsail_WorkDescr * desc)
{
#if 0
  //cg.Clear(); // start and stop to zero too
#endif
  // threadsInParallelTeam does not need to be init (done in start parallel)
  desc->hasCancel = FALSE;
}

////////////////////////////////////////////////////////////////////////////////
// Team Descriptor
////////////////////////////////////////////////////////////////////////////////

INLINE void TeamDescr_InitTeamDescr(
    __MSPACE omptarget_hsail_TeamDescr * desc)
{
  PRINT0(LD_IO, "call init\n");
  TaskDescr_InitLevelZeroTaskDescr(&desc->levelZeroTaskDescr);
  WorkDescr_InitWorkDescr(&desc->workDescrForActiveParallel);
#if 0
  //omp_init_lock(criticalLock);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Get private data structure for thread
////////////////////////////////////////////////////////////////////////////////

// Utility routines for HSA threads
INLINE __MSPACE omptarget_hsail_TeamDescr * getMyTeamDescriptor()
{
  return &omptarget_hsail_teamPrivateContext.team[GetOmpTeamId()];
}

INLINE __MSPACE omptarget_hsail_TeamDescr * getMyTeamDescriptorById(int globalTeamId)
{
  return &omptarget_hsail_teamPrivateContext.team[globalTeamId];
}

INLINE __MSPACE omptarget_hsail_WorkDescr * getMyWorkDescriptor()
{
  __MSPACE omptarget_hsail_TeamDescr * currTeamDescr = getMyTeamDescriptor();
  return &currTeamDescr->workDescrForActiveParallel;
}

INLINE __MSPACE omptarget_hsail_WorkDescr * getMyWorkDescriptorById(int globalTeamId)
{
  __MSPACE omptarget_hsail_TeamDescr * currTeamDescr = getMyTeamDescriptorById(globalTeamId);
  return &currTeamDescr->workDescrForActiveParallel;
}

INLINE __MSPACE omptarget_hsail_TaskDescr * getMyTopTaskDescriptorById(int globalThreadId)
{
  return ThreadPrivateContext_GetTopLevelTaskDescr(&omptarget_hsail_threadPrivateContext, globalThreadId);
}

INLINE __MSPACE omptarget_hsail_TaskDescr * getMyTopTaskDescriptor()
{
  return getMyTopTaskDescriptorById(GetGlobalThreadId());
}


