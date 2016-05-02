//===------------ option.h - HSAIL OpenMP GPU options ------------ HSA -*-===//
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
// GPU default options
//
//===----------------------------------------------------------------------===//

#ifndef _OPTION_H_
#define _OPTION_H_

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// following two defs must match absolute limit hardwired in the host RTL

// Limit this size to phisical units for now, this should be way bigger
/* Use this to decide total teams: active groups * number of compute unit */
#define TEAMS_ABSOLUTE_LIMIT 4*8 /* omptx limit (must match teamsAbsoluteLimit) */
// Let's don't simulate SIMD yet, WAVEFRONTSIZE X THREAD_ABSOLUTE_LIMIT is the local group size
#define THREAD_ABSOLUTE_LIMIT 2 /* omptx limit (must match threadAbsoluteLimit) */

// max number of blocks depend on the kernel we are executing - pick default here
#define MAX_NUM_TEAMS TEAMS_ABSOLUTE_LIMIT
#define WAVEFRONTSIZE 64
#define MAX_NUM_WAVES MAX_NUM_TEAMS * THREAD_ABSOLUTE_LIMIT

#ifdef OMPTHREAD_IS_WAVEFRONT
#define MAX_NUM_THREADS MAX_NUM_WAVES
#else
#define MAX_NUM_THREADS MAX_NUM_WAVES * WAVEFRONTSIZE
#endif

#ifdef OMPTHREAD_IS_WAVEFRONT
  // assume here one OpenMP thread per HSA wavefront
  #define MAX_NUM_OMP_THREADS MAX_NUM_WAVES
#else
  // assume here one OpenMP thread per HSA thread
  #define MAX_NUM_OMP_THREADS MAX_NUM_THREADS
#endif

////////////////////////////////////////////////////////////////////////////////
// algo options
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// data options
////////////////////////////////////////////////////////////////////////////////


// decide if counters are 32 or 64 bit
#define Counter unsigned long

// aee: KMP defines kmp_int to be 32 or 64 bits depending on the target.
// think we don't need it here (meaning we can be always 64 bit compatible)
/*
#ifdef KMP_I8
  typedef kmp_int64		kmp_int;
#else
  typedef kmp_int32		kmp_int;
#endif
*/

////////////////////////////////////////////////////////////////////////////////
// misc options (by def everythig here is device)
////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
  #define EXTERN extern "C"
#else
  #define EXTERN extern
#endif

#define INLINE __inline__
#define NOINLINE //__noinline__
#ifndef TRUE
  #define TRUE 1
#endif
#ifndef FALSE
  #define FALSE 0
#endif

#endif
