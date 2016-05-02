//===------------- debug.h - HSAIL OpenMP debug macros ----------- HSA -*-===//
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
// This file contains debug macros to be used in the application.
//
//   Usage guide
//
//   PRINT0(flag, str)        : if debug flag is on, print (no arguments)
//   PRINT(flag, str, args)   : if debug flag is on, print (arguments)
//   DON(flag)                : return true if debug flag is on
//
//   ASSERT(flag, cond, str, args): if test flag is on, test the condition
//                                  if the condition is false, print str+args
//          and assert.
//          CAUTION: cond may be evaluate twice
//   AON(flag)                     : return true if test flag is on
//
//   WARNING(flag, str, args)      : if warning flag is on, print the warning
//   WON(flag)                     : return true if warning flag is on
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_HSAIL_DEBUG_H_
#define _OMPTARGET_HSAIL_DEBUG_H_

////////////////////////////////////////////////////////////////////////////////
// set desired level of debugging
////////////////////////////////////////////////////////////////////////////////

#define LD_SET_NONE               0ULL /* none */
#define LD_SET_ALL               -1ULL /* all */

// pos 1
#define LD_SET_LOOP             0x1ULL /* basic loop */
#define LD_SET_LOOPD            0x2ULL /* basic loop */
#define LD_SET_PAR              0x4ULL /* basic parallel */
#define LD_SET_PARD             0x8ULL /* basic parallel */

// pos 2
#define LD_SET_SYNC            0x10ULL /* sync info */
#define LD_SET_SYNCD           0x20ULL /* sync info */
#define LD_SET_WAIT            0x40ULL /* state when waiting */
#define LD_SET_TASK            0x80ULL /* print task info (high level) */

// pos 3
#define LD_SET_IO             0x100ULL /* big region io (excl atomic) */
#define LD_SET_IOD            0x200ULL /* big region io (excl atomic) */
#define LD_SET_ENV            0x400ULL /* env info */
#define LD_SET_CANCEL         0x800ULL /* print cancel info */

// pos 4
#define LD_SET_MEM           0x1000ULL /* malloc / free */



////////////////////////////////////////////////////////////////////////////////
// set the desired flags to print selected output

//#define OMPTARGET_HSAIL_DEBUG (LD_SET_ALL)
//#define OMPTARGET_HSAIL_DEBUG (LD_SET_LOOP) // limit to loop printfs to save on HSA buffer
//#define OMPTARGET_HSAIL_DEBUG (LD_SET_IO)
//#define OMPTARGET_HSAIL_DEBUG (LD_SET_IO | LD_SET_ENV)
//#define OMPTARGET_HSAIL_DEBUG (LD_SET_PAR)

#ifndef OMPTARGET_HSAIL_DEBUG
  #define OMPTARGET_HSAIL_DEBUG LD_SET_NONE
#elif OMPTARGET_HSAIL_DEBUG
  #warning debug is used, not good for measurements
#endif

////////////////////////////////////////////////////////////////////////////////
// set desired level of asserts
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// available flags

#define LT_SET_NONE       0x0 /* unsafe */
#define LT_SET_SAFETY     0x1 /* check malloc type of stuff, input at creation, cheap */
#define LT_SET_INPUT      0x2 /* check also all runtime inputs */
#define LT_SET_FUSSY      0x4 /* fussy checks, expensive */

////////////////////////////////////////////////////////////////////////////////
// set the desired flags

#ifndef OMPTARGET_HSAIL_TEST
  #if OMPTARGET_HSAIL_DEBUG
    #define OMPTARGET_HSAIL_TEST (LT_SET_FUSSY)
  #else
    #define OMPTARGET_HSAIL_TEST (LT_SET_SAFETY)
  #endif
#endif

////////////////////////////////////////////////////////////////////////////////
// set desired level of warnings
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// available flags

#define LW_SET_ALL         -1
#define LW_SET_NONE       0x0
#define LW_SET_ENV        0x1
#define LW_SET_INPUT      0x2
#define LW_SET_FUSSY      0x4

////////////////////////////////////////////////////////////////////////////////
// set the desired flags

#if OMPTARGET_HSAIL_DEBUG
  #define OMPTARGET_HSAIL_WARNING  (LW_SET_NONE)
#else
  #define OMPTARGET_HSAIL_WARNING (LW_SET_FUSSY)
#endif


////////////////////////////////////////////////////////////////////////////////
// implemtation for debug
////////////////////////////////////////////////////////////////////////////////

#if OMPTARGET_HSAIL_DEBUG || OMPTARGET_HSAIL_TEST || OMPTARGET_HSAIL_WARNING
//  #include <stdio.h>
#endif
#if OMPTARGET_HSAIL_TEST
//  #include <assert.h>
#endif

// set flags that are tested (inclusion properties)

#define LD_ALL      (LD_SET_ALL)

#define LD_LOOP     (LD_SET_LOOP | LD_SET_LOOPD)
#define LD_LOOPD    (LD_SET_LOOPD)
#define LD_PAR      (LD_SET_PAR  | LD_SET_PARD)
#define LD_PARD     (LD_SET_PARD)

// pos 2
#define LD_SYNC     (LD_SET_SYNC  | LD_SET_SYNCD)
#define LD_SYNCD    (LD_SET_SYNCD)
#define LD_WAIT     (LD_SET_WAIT)
#define LD_TASK     (LD_SET_TASK)

// pos 3
#define LD_IO       (LD_SET_IO | LD_SET_IOD)
#define LD_IOD      (LD_SET_IOD)
#define LD_ENV      (LD_SET_ENV)
#define LD_CANCEL   (LD_SET_CANCEL)

// pos 3
#define LD_MEM      (LD_SET_MEM)

#define WAVEFRONTSIZE 64

// implement
#if OMPTARGET_HSAIL_DEBUG

  #define DON(_flag) ((OMPTARGET_HSAIL_DEBUG) & (_flag))

  #define PRINT0(_flag, _str) { if (DON(_flag)) { \
    printf("[%s:%d] <g %2d, t %4d, w %2d, l %2d>: " _str, __FILE__, __LINE__, get_group_id(0), get_local_id(0), \
      get_local_id(0) / WAVEFRONTSIZE, get_local_id(0) & 0x1F); }}

  #define PRINT(_flag, _str, _args...) { if (DON(_flag)) { \
    printf("[%s:%d] <g %2d, t %4d, w %2d, l %2d>: " _str, __FILE__, __LINE__, get_group_id(0), get_local_id(0), \
      get_local_id(0) / WAVEFRONTSIZE, get_local_id(0) & 0x1F, _args); }}
#else

  #define DON(_flag) (FALSE)
  #define PRINT0(flag, str)
  #define  PRINT(flag, str, _args...)

#endif

// for printing without worring about precision, pointers...
#define P64(_x) ((unsigned long long)(_x))

////////////////////////////////////////////////////////////////////////////////
// early defs for test
////////////////////////////////////////////////////////////////////////////////

#define LT_SAFETY     (LT_SET_SAFETY | LT_SET_INPUT | LT_SET_FUSSY)
#define LT_INPUT      (LT_SET_INPUT | LT_SET_FUSSY)
#define LT_FUSSY      (LT_SET_FUSSY)

#if OMPTARGET_HSAIL_DEBUG
  #define TON(_flag) ((OMPTARGET_HSAIL_TEST) & (_flag))
  #define ASSERT0(_flag, _cond, _str) { if (TON(_flag) && !(_cond)) { \
    printf("<g %3d, t %4d, w %2d, l %2d> ASSERT: " _str "\n", get_group_id(0), get_local_id(0), \
      get_local_id(0) / WAVEFRONTSIZE, get_local_id(0) & 0x1F); }}
  #define ASSERT(_flag, _cond, _str, _args...) { if (TON(_flag) && !(_cond)) { \
    printf("<g %3d, t %4d, w %2d, l %d2> ASSERT: " _str "\n", get_group_id(0), get_local_id(0), \
      get_local_id(0) / WAVEFRONTSIZE, get_local_id(0) & 0x1F, _args); }}

#else

  #define TON(_flag) (FALSE)
  #define ASSERT0(_flag, _cond, _str)
  #define  ASSERT(_flag, _cond, _str, _args...)

#endif


////////////////////////////////////////////////////////////////////////////////
// early defs for warning

#define LW_ALL        (LW_SET_ALL)
#define LW_ENV        (LW_SET_FUSSY | LW_SET_INPUT | LW_SET_ENV)
#define LW_INPUT      (LW_SET_FUSSY | LW_SET_INPUT)
#define LW_FUSSY      (LW_SET_FUSSY)

#if OMPTARGET_HSAIL_DEBUG

  #define WON(_flag) ((OMPTARGET_HSAIL_WARNING) & (_flag))
  #define WARNING0(_flag, _str) { if (WON(_flag)) { \
    printf("<g %2d, t %4d, w %2d, l %2d> WARNING: " _str, get_group_id(0), get_local_id(0), \
      get_local_id(0) / WAVEFRONTSIZE, get_local_id(0) & 0x1F); }}
  #define WARNING(_flag, _str, _args...) { if (WON(_flag)) { \
    printf("<g %2d, t %4d, w %2d, l %2d> WARNING: " _str,  get_group_id(0), get_local_id(0), \
      get_local_id(0) / WAVEFRONTSIZE, get_local_id(0) & 0x1F, _args); }}

#else

  #define WON(_flag) (FALSE)
  #define WARNING0(_flag, _str)
  #define  WARNING(_flag, _str, _args...)

#endif


#endif
