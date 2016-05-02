//===------------ loop.cl - hsail OpenMP loop constructs --------- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the KMPC interface
// for the loop construct plus other worksharing constructs that use the same
// interface as loops.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions do not need to be inside loopi.h for multiple includes
////////////////////////////////////////////////////////////////////////////////

INLINE static int OrderedSchedule(kmp_sched_t schedule)
{
  return schedule >= kmp_sched_ordered_first &&
    schedule <= kmp_sched_ordered_last;
}

INLINE static void dispatch_fini()
{
  // nothing
}

////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (dyn loops)
////////////////////////////////////////////////////////////////////////////////

// init
EXTERN void __kmpc_dispatch_init_4 (
        kmp_Indent * loc,
        int32_t gtid,
        int32_t schedule,
        int32_t lb,
        int32_t ub,
        int32_t st,
        int32_t chunk)
{
  PRINT0(LD_IO, "call kmpc_dispatch_init_4\n");
#if 0
  dispatch_init(int32_t, int32_t) ((kmp_sched_t) schedule, lb, ub, st, chunk);
#endif
}

EXTERN void __kmpc_dispatch_init_4u(
        kmp_Indent * loc,
        int32_t gtid,
        int32_t schedule,
        uint32_t lb,
        uint32_t ub,
        int32_t st,
        int32_t chunk)
{
  PRINT0(LD_IO, "call kmpc_dispatch_init_4u\n");
#if 0
  dispatch_init(uint32_t, int32_t)((kmp_sched_t) schedule, lb, ub, st, chunk);
#endif
}

EXTERN void __kmpc_dispatch_init_8(
        kmp_Indent * loc,
        int32_t gtid,
        int32_t schedule,
        int64_t lb,
        int64_t ub,
        int64_t st,
        int64_t chunk)
{
  PRINT0(LD_IO, "call kmpc_dispatch_init_8\n");
#if 0
  dispatch_init(int64_t, int64_t)((kmp_sched_t) schedule, lb, ub, st, chunk);
#endif
}

EXTERN void __kmpc_dispatch_init_8u(
        kmp_Indent * loc,
        int32_t gtid,
        int32_t schedule,
        uint64_t lb,
        uint64_t ub,
        int64_t st,
        int64_t chunk)
{
  PRINT0(LD_IO, "call kmpc_dispatch_init_8u\n");
#if 0
  dispatch_init(uint64_t, int64_t)((kmp_sched_t) schedule, lb, ub, st, chunk);
#endif
}

// next
EXTERN int __kmpc_dispatch_next_4(
        kmp_Indent * loc,
        int32_t gtid,
        int32_t * p_last,
        int32_t * p_lb,
        int32_t * p_ub,
        int32_t * p_st)
{
  PRINT0(LD_IO, "call kmpc_dispatch_next_4\n");
#if 0
  return <int32_t, int32_t>::dispatch_next(p_last, p_lb, p_ub, p_st);
#endif
}

EXTERN int __kmpc_dispatch_next_4u(
        kmp_Indent * loc,
        int32_t gtid,
        int32_t * p_last,
        uint32_t * p_lb,
        uint32_t * p_ub,
        int32_t * p_st)
{
  PRINT0(LD_IO, "call kmpc_dispatch_next_4u\n");
#if 0
  return <uint32_t, int32_t>::dispatch_next(p_last, p_lb, p_ub, p_st);
#endif
}

EXTERN int __kmpc_dispatch_next_8(
        kmp_Indent * loc,
        int32_t gtid,
        int32_t * p_last,
        int64_t * p_lb,
        int64_t * p_ub,
        int64_t * p_st)
{
  PRINT0(LD_IO, "call kmpc_dispatch_next_8\n");
#if 0
  return <int64_t, int64_t>::dispatch_next(p_last, p_lb, p_ub, p_st);
#endif
}

EXTERN int __kmpc_dispatch_next_8u(kmp_Indent * loc, int32_t gtid, 
  int32_t * p_last, uint64_t * p_lb, uint64_t * p_ub, int64_t * p_st)
{
  PRINT0(LD_IO, "call kmpc_dispatch_next_8u\n");
#if 0
  return <uint64_t, int64_t>::dispatch_next(p_last, p_lb, p_ub, p_st);
#endif
}

// dynamic fini
EXTERN void __kmpc_dispatch_fini_4 (kmp_Indent * loc, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_dispatch_fini_4\n");

  dispatch_fini();
}

EXTERN void __kmpc_dispatch_fini_4u (kmp_Indent * loc, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_dispatch_fini_4u\n");

  dispatch_fini();
}

EXTERN void __kmpc_dispatch_fini_8 (kmp_Indent * loc, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_dispatch_fini_8\n");

  dispatch_fini();
}

EXTERN void __kmpc_dispatch_fini_8u (kmp_Indent * loc, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_dispatch_fini_8u\n");

  dispatch_fini();
}


////////////////////////////////////////////////////////////////////////////////
// KMP interface implementation (static loops)
////////////////////////////////////////////////////////////////////////////////

#define LOOPTYPE int32_t
#define CHNKTYPE int32_t
#include "loopi.h"
#undef CHNKTYPE //int32_t
#undef LOOPTYPE //int32_t

EXTERN void __kmpc_for_static_init_4(
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t schedtype,
        __private int32_t *plastiter,
        __private int32_t *plower,
        __private int32_t *pupper,
        __private int32_t *pstride,
        int32_t incr,
        int32_t chunk)
{
    PRINT0(LD_IO, "call kmpc_for_static_init_4\n");
    for_static_init_int32_t_int32_t (schedtype, plower, pupper, pstride, chunk);
}

#define LOOPTYPE uint32_t
#define CHNKTYPE int32_t
#include "loopi.h"
#undef CHNKTYPE //int32_t
#undef LOOPTYPE //uint32_t

EXTERN void __kmpc_for_static_init_4u (
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t schedtype,
        __private int32_t *plastiter,
        __private uint32_t *plower,
        __private uint32_t *pupper,
        __private int32_t *pstride,
        int32_t incr,
        int32_t chunk)
{
    PRINT0(LD_IO, "call kmpc_for_static_init_4u\n");
    for_static_init_uint32_t_int32_t (schedtype, plower, pupper, pstride, chunk);
}

#define LOOPTYPE int64_t
#define CHNKTYPE int64_t
#include "loopi.h"
#undef CHNKTYPE //int64_t
#undef LOOPTYPE //int64_t

EXTERN void __kmpc_for_static_init_8(
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t schedtype,
        __private int32_t *plastiter,
        __private int64_t *plower,
        __private int64_t *pupper,
        __private int64_t *pstride,
        int64_t incr,
        int64_t chunk)
{
    PRINT0(LD_IO, "call kmpc_for_static_init_8\n");
    for_static_init_int64_t_int64_t (schedtype, plower, pupper, pstride, chunk);
}

#define LOOPTYPE uint64_t
#define CHNKTYPE int64_t
#include "loopi.h"
#undef CHNKTYPE //int64_t
#undef LOOPTYPE //uint64_t

EXTERN void __kmpc_for_static_init_8u (
        __private kmp_Indent *loc,
        int32_t global_tid,
        int32_t schedtype,
        __private int32_t *plastiter,
        __private uint64_t *plower,
        __private uint64_t *pupper,
        __private int64_t *pstride,
        int64_t incr,
        int64_t chunk)
{
    PRINT0(LD_IO, "call kmpc_for_static_init_8u\n");
    for_static_init_uint64_t_int64_t(schedtype, plower, pupper, pstride, chunk);
}

// static fini
EXTERN void __kmpc_for_static_fini(
        __private kmp_Indent *loc,
        int32_t global_tid)
{
    PRINT0(LD_IO, "call kmpc_for_static_fini\n");
}

