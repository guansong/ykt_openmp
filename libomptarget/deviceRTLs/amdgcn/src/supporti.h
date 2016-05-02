//===--------- supporti.h - HSAIL OpenMP support functions ------- HSA -*-===//
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
// Wrapper implementation to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// support: get info from machine
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// machine: get number of (assuming 1D layout)

INLINE int GetNumberOfThreadsInBlock()
{
  //return blockDim.x;
  return get_local_size(0);
}

INLINE int GetNumberOfWaveFrontsInBlock()
{
  ASSERT(LT_FUSSY, GetNumberOfThreadsInBlock() % WAVEFRONTSIZE == 0,
    "expected threads num %d to be a multiple of warp size %d\n",
    GetNumberOfThreadsInBlock(), WAVEFRONTSIZE);
   return GetNumberOfThreadsInBlock() / WAVEFRONTSIZE;
}

INLINE int GetNumberOfBlocksInKernel()
{
  //return gridDim.x;
  return get_num_groups(0);
}


////////////////////////////////////////////////////////////////////////////////
// machine: get ids  (assuming 1D layout)

INLINE int GetThreadIdInBlock()
{
  //return threadIdx.x;
  return get_local_id(0);
}

INLINE int GetWaveFrontIdInBlock()
{
  ASSERT(LT_FUSSY, GetNumberOfThreadsInBlock() % WAVEFRONTSIZE == 0,
    "expected threads num %d to be a multiple of warp size %d\n",
    GetNumberOfThreadsInBlock(), WAVEFRONTSIZE);
  return  GetThreadIdInBlock() / WAVEFRONTSIZE;
}

INLINE int GetBlockIdInKernel()
{
  //return blockIdx.x;
  return get_group_id(0);
}

////////////////////////////////////////////////////////////////////////////////
// Global thread id used to locate thread info

INLINE int GetGlobalThreadId()
{
  #ifdef OMPTHREAD_IS_WAVEFRONT
    return GetBlockIdInKernel() * GetNumberOfWaveFrontsInBlock() + GetWaveFrontIdInBlock();
  #else
    return GetBlockIdInKernel() * GetNumberOfThreadsInBlock() + GetThreadIdInBlock();
    // return get_global_id(0);
  #endif
}

INLINE int GetNumberOfGlobalThreadIds()
{
  #ifdef OMPTHREAD_IS_WAVEFRONT
    return GetNumberOfWaveFrontsInBlock() * GetNumberOfBlockInKernel();
  #else
    return GetNumberOfThreadsInBlock() * GetNumberOfBlocksInKernel();
    //return get_global_size(0);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
// global  team id used to locate team info

INLINE int GetGlobalTeamId()
{
  return GetBlockIdInKernel();
}

INLINE int GetNumberOfGlobalTeamIds()
{
  return GetNumberOfBlocksInKernel();
}

////////////////////////////////////////////////////////////////////////////////
// OpenMP Thread id linked to OpenMP

INLINE int GetOmpThreadId(int globalThreadId)
{
  // omp_thread_num
  __MSPACE omptarget_hsail_TaskDescr *currTaskDescr =
      ThreadPrivateContext_GetTopLevelTaskDescr(&omptarget_hsail_threadPrivateContext, globalThreadId);

  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");

  int rc = currTaskDescr->data.items.threadId;
  return rc;
}

INLINE int GetNumberOfOmpThreads(int globalThreadId)
{
  // omp_num_threads
  __MSPACE omptarget_hsail_TaskDescr *currTaskDescr =
      ThreadPrivateContext_GetTopLevelTaskDescr(&omptarget_hsail_threadPrivateContext, globalThreadId);

  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");

  int rc = currTaskDescr->data.items.threadsInTeam;
  return rc;
}

////////////////////////////////////////////////////////////////////////////////
// Team id linked to OpenMP

INLINE int GetOmpTeamId()
{
  // omp_team_num
  return GetGlobalTeamId(); // assume 1 block per team
}

INLINE int GetNumberOfOmpTeams()
{
  // omp_num_teams
  return GetNumberOfGlobalTeamIds(); // assume 1 block per team
}


////////////////////////////////////////////////////////////////////////////////
// get OpenMP number of procs

INLINE int GetNumberOfProcsInTeam()
{
  #ifdef OMPTHREAD_IS_WAVEFRONT
    return GetNumberOfWaveFrontsInBlock();
  #else
    return GetNumberOfThreadsInBlock();
  #endif
}


////////////////////////////////////////////////////////////////////////////////
// Masters

INLINE int IsTeamMaster(int ompThreadId)
{
  return (ompThreadId == 0);
}

INLINE int IsWaveFrontMaster(int ompThreadId)
{
  return (ompThreadId % WAVEFRONTSIZE == 0);
}

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////

#if 0
INLINE unsigned long PadBytes(
  unsigned long size,
  unsigned long alignment) // must be a power of 2
{
  // compute the necessary padding to satify alignment constraint
  ASSERT(LT_FUSSY, (alignment & (alignment - 1)) == 0,
    "alignment %ld is not a power of 2\n", alignment);
  return (~(unsigned long) size + 1) & (alignment - 1);
}

INLINE void *SafeMalloc(size_t size, const char *msg) // check if success
{
  void * ptr = malloc(size);
  PRINT(LD_MEM, "malloc data of size %d for %s: 0x%llx\n", size, msg,
    P64(ptr));
  ASSERT(LT_SAFETY, ptr, "failed to allocate %d bytes for %s\n", size, msg);
  return ptr;
}

INLINE void *SafeFree(void *ptr, const char *msg)
{
  PRINT(LD_MEM, "free data ptr 0x%llx for %s\n", P64(ptr), msg);
  free(ptr);
  return NULL;
}
#endif
