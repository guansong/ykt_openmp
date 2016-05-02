//===------ cancel.cl - hsail OpenMP cancel interface ------------ HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to be used in the implementation of OpenMP cancel.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"

EXTERN int32_t __kmpc_cancellationpoint(
        kmp_Indent* loc,
        int32_t global_tid,
        int32_t cancelVal)
{
    PRINT(LD_IO, "call kmpc_cancellationpoint(cancel val %d)\n", cancelVal);
    // disabled
    return FALSE;
}

EXTERN int32_t __kmpc_cancel(
        kmp_Indent* loc,
        int32_t global_tid,
        int32_t cancelVal)
{
    PRINT(LD_IO, "call kmpc_cancel(cancel val %d)\n", cancelVal);
    // disabled
    return FALSE;
}
