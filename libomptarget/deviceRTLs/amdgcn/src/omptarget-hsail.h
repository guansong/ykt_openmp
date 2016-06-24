//===---- omptarget-hsail.h - HSAIL OpenMP GPU initialization ---- HSA -*-===//
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
// This file contains the declarations of all library macros, types,
// and functions.
//
//===----------------------------------------------------------------------===//

#ifndef __OMPTARGET_HSAIL_H
#define __OMPTARGET_HSAIL_H

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

// std includes
//#include <stdlib.h>
//#include <stdint.h>

#ifndef _WITHOUT_STDDEF_HEADER_
#include <stddef.h>
#endif

typedef char int8_t;
typedef unsigned char uint8_t;

typedef short int16_t;
typedef unsigned short uint16_t;

typedef int int32_t;
typedef unsigned int uint32_t;

typedef long int64_t;
typedef unsigned long uint64_t;

// opencl includes
#if __OPENCL_VERSION__ >= 200
#define __MSPACE __global
#else
// this is only for syntax checking in OpenCL 1.x
#define __MSPACE __constant
#endif

// local includes
#include "option.h"         // choices we have
#include "interface.h"      // interfaces with omp, compiler, and user
#include "debug.h"          // debug

#include "support.h"
#include "counter_group.h"

#define OMPTARGET_HSAIL_VERSION 1.1

// used by the library for the interface with the app
#define DISPATCH_FINISHED 0
#define DISPATCH_NOTFINISHED 1

// used by dynamic scheduling
#define FINISHED 0
#define NOT_FINISHED 1
#define LAST_CHUNK 2

#define TEAM_MASTER 0
#define BARRIER_COUNTER 0
#define ORDERED_COUNTER 1

// For Intrinsic mapping
#if OMPTARGET_HSAIL_DEBUG == 0
// Do not use OpenCL builtin functions
#define get_global_size _Z15get_global_sizej
EXTERN long _Z15get_global_sizej(int);

#define get_global_id _Z13get_global_idj
EXTERN long _Z13get_global_idj(int);

#define get_local_size _Z14get_local_sizej
EXTERN long _Z14get_local_sizej(int);

#define get_local_id _Z12get_local_idj
EXTERN long _Z12get_local_idj(int);

#define get_num_groups _Z14get_num_groupsj
EXTERN long _Z14get_num_groupsj(int);

#define get_group_id _Z12get_group_idj
EXTERN long _Z12get_group_idj(int);

#define barrier _Z7barrierj
EXTERN void _Z7barrierj(int);

#define atomic_compare_exchange_strong __compare_exchange_int_global
EXTERN int __compare_exchange_int_global(__global omp_lock_t *, int *, int);

#define atomic_store __exchange_int_global
EXTERN int __exchange_int_global(__global omp_lock_t *, int);

#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE 1
#endif
#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE 2
#endif

#else
// Use the OpenCL builtin functions
#endif


////////////////////////////////////////////////////////////////////////////////
// global ICV

typedef struct omptarget_hsail_GlobalICV {
  double  gpuCycleTime; // currently statically determined, should be set by host
  uint8_t cancelPolicy; // 1 bit: enabled (true) or disabled (false)
} omptarget_hsail_GlobalICV;

////////////////////////////////////////////////////////////////////////////////
// task ICV and (implicit & explicit) task state
#if 0
static const uint8_t TaskDescr_SchedMask   = (0x1 | 0x2 | 0x4);
static const uint8_t TaskDescr_IsDynamic   = 0x8;
static const uint8_t TaskDescr_InPar       = 0x10;
static const uint8_t TaskDescr_IsParConstr = 0x20;
#endif

#define TaskDescr_SchedMask   (0x1 | 0x2 | 0x4)
#define TaskDescr_IsDynamic   0x8
#define TaskDescr_InPar       0x10
#define TaskDescr_IsParConstr 0x20

typedef struct omptarget_hsail_TaskDescr omptarget_hsail_TaskDescr;
struct omptarget_hsail_TaskDescr {
  union { // both have same size
    uint64_t vect[3];
    struct TaskDescr_items {
      uint8_t  flags; // 6 bit used (see flag above)
      uint8_t  unused;
      uint16_t nthreads; // thread num for subsequent parallel regions
      uint16_t threadlimit; // thread limit ICV
      uint16_t threadId; // thread id
      uint16_t threadsInTeam; // threads in current team
      uint64_t runtimeChunkSize; // runtime chunk size
    } items;
  } data;
  __MSPACE omptarget_hsail_TaskDescr *prev;
};

////////////////////////////////////////////////////////////////////////////////
// build on kmp
typedef struct omptarget_hsail_ExplicitTaskDescr {
  omptarget_hsail_TaskDescr taskDescr; // omptarget_hsail task description (must be first)
  kmp_TaskDescr   kmpTaskDescr; // kmp task description (must be last)
} omptarget_hsail_ExplicitTaskDescr;


////////////////////////////////////////////////////////////////////////////////
// thread private data (struct of arrays for better coalescing)
// tid refers here to the global thread id
// do not support multiple concurrent kernel a this time

typedef struct omptarget_hsail_ThreadPrivateContext {
  // task ICV for implict threads in the only parallel region
  omptarget_hsail_TaskDescr levelOneTaskDescr[MAX_NUM_OMP_THREADS];
  // pointer where to find the current task ICV (top of the stack)
  __MSPACE omptarget_hsail_TaskDescr *topLevelTaskDescrPtr[MAX_NUM_OMP_THREADS];
  // parallel
  uint16_t nthreadsForNextPar[MAX_NUM_OMP_THREADS];
  // sync
  Counter priv[MAX_NUM_OMP_THREADS];
  // schedule (for dispatch)
  kmp_sched_t schedule[MAX_NUM_OMP_THREADS]; // remember schedule type for #for
  int64_t chunk[MAX_NUM_OMP_THREADS];
  int64_t loopUpperBound[MAX_NUM_OMP_THREADS];
  // state for dispatch with dyn/guided OR static (never use both at a time)
  Counter currEvent_or_nextLowerBound[MAX_NUM_OMP_THREADS];
  Counter eventsNum_or_stride[MAX_NUM_OMP_THREADS];
} omptarget_hsail_ThreadPrivateContext;


////////////////////////////////////////////////////////////////////////////////
// Descriptor of a parallel region (worksharing in general)

typedef struct omptarget_hsail_WorkDescr {
  omptarget_hsail_CounterGroup counters; // for barrier (no other needed)
  omptarget_hsail_TaskDescr masterTaskICV;
  bool hasCancel;
} omptarget_hsail_WorkDescr;


////////////////////////////////////////////////////////////////////////////////
// team private data (struct of arrays for better coalescing)

typedef struct omptarget_hsail_TeamDescr {
  omptarget_hsail_TaskDescr levelZeroTaskDescr ; // icv for team master initial thread
  omptarget_hsail_WorkDescr workDescrForActiveParallel; // one, ONLY for the active par
  omp_lock_t criticalLock;
} omptarget_hsail_TeamDescr;

typedef struct omptarget_hsail_TeamPrivateContext {
  omptarget_hsail_TeamDescr team[MAX_NUM_TEAMS];
} omptarget_hsail_TeamPrivateContext;


////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

extern __MSPACE omptarget_hsail_TeamPrivateContext omptarget_hsail_teamPrivateContext;
extern __MSPACE omptarget_hsail_ThreadPrivateContext omptarget_hsail_threadPrivateContext;
extern __MSPACE omptarget_hsail_GlobalICV omptarget_hsail_globalICV;

////////////////////////////////////////////////////////////////////////////////
// get private data structures
////////////////////////////////////////////////////////////////////////////////

INLINE __MSPACE omptarget_hsail_TeamDescr * getMyTeamDescriptor(void);
INLINE __MSPACE omptarget_hsail_TeamDescr * getMyTeamDescriptorById(int globalTeamId);

INLINE __MSPACE omptarget_hsail_WorkDescr * getMyWorkDescriptor(void);
INLINE __MSPACE omptarget_hsail_WorkDescr * getMyWorkDescriptorById(int globalTeamId);

INLINE __MSPACE omptarget_hsail_TaskDescr * getMyTopTaskDescriptor(void);
INLINE __MSPACE omptarget_hsail_TaskDescr * getMyTopTaskDescriptorById(int globalThreadId);


////////////////////////////////////////////////////////////////////////////////
// inlined implementation
////////////////////////////////////////////////////////////////////////////////
#include "omptarget-hsaili.h"
#include "supporti.h"
#include "counter_groupi.h"

#define PRINTTASKDESCR(_str, taskDescr) {\
  omp_sched_t sched = (taskDescr->data.items.flags & TaskDescr_SchedMask) + 1; \
  PRINT(LD_ALL, _str " task %p: isP %x, inP %x, dyn %d, sched %d, nthds %d, tlt %d, tid %d, team %d, chunk %ld (%p)\n", \
      taskDescr, \
      (taskDescr->data.items.flags & TaskDescr_IsParConstr)/TaskDescr_IsParConstr, \
      (taskDescr->data.items.flags & TaskDescr_InPar)/TaskDescr_InPar, \
      (taskDescr->data.items.flags & TaskDescr_IsDynamic)/TaskDescr_IsDynamic, \
      sched, \
      (int)taskDescr->data.items.nthreads, \
      (int)taskDescr->data.items.threadlimit, \
      (int)taskDescr->data.items.threadId, \
      (int)taskDescr->data.items.threadsInTeam, \
      taskDescr->data.items.runtimeChunkSize, \
      taskDescr->prev); \
}

INLINE void PrintTaskDescr(
    __MSPACE omptarget_hsail_TaskDescr *taskDescr,
    __constant char *title,
    int level);

#endif
