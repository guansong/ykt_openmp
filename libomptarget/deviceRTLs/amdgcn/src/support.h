//===--------- support.h - HSAIL support functions -------- HSA -*-===//
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
// Wrapper to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// get info from machine
////////////////////////////////////////////////////////////////////////////////

// get global ids to locate tread/team info (constant regardless of OMP)
INLINE int GetGlobalThreadId(void);
INLINE int GetGlobalTeamId(void);

// get global number of ids to size thread/team data structures
INLINE int GetNumberOfGlobalThreadIds(void);
INLINE int GetNumberOfGlobalTeamIds(void);

// get OpenMP thread and team ids
INLINE int GetOmpThreadId(int globalThreadId);        // omp_thread_num
INLINE int GetOmpTeamId(void);                            // omp_team_num

// get OpenMP number of threads and team
INLINE int GetNumberOfOmpThreads(int globalThreadId); // omp_num_threads
INLINE int GetNumberOfOmpTeams(void);                     // omp_num_teams

// get OpenMP number of procs
INLINE int GetNumberOfProcsInTeam(void);

// masters
INLINE int IsTeamMaster(int ompThreadId);
INLINE int IsWarpMaster(int ompThreadId);


// get low level ids of resources
INLINE int GetThreadIdInBlock(void);
INLINE int GetWarpIdInBlock(void);
INLINE int GetBlockIdInKernel(void);

// Get low level number of resources
INLINE int GetNumberOfThreadsInBlock(void);
INLINE int GetNumberOfWarpsInBlock(void);
INLINE int GetNumberOfBlocksInKernel(void);

////////////////////////////////////////////////////////////////////////////////
// Memory
////////////////////////////////////////////////////////////////////////////////


// safe alloc and free
INLINE void *SafeMalloc(size_t size, const char *msg); // check if success
INLINE void *SafeFree(void *ptr, const char *msg);
// pad to a alignment (power of 2 only)
INLINE unsigned long PadBytes(unsigned long size, unsigned long alignment);
#define ADD_BYTES(_addr, _bytes) ((void *)((char *)((void *)(_addr))+(_bytes)))
#define SUB_BYTES(_addr, _bytes) ((void *)((char *)((void *)(_addr))-(_bytes)))
