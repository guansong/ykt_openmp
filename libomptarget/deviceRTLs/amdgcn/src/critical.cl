//===------ critical.cl - hsail OpenMP critical ------------------ HSA -*-===//
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
// This file contains the implementation of critical with KMPC interface
//
//===----------------------------------------------------------------------===//

//#include <stdio.h>
//#include <complex.h>

#include "omptarget-hsail.h"

EXTERN void __kmpc_critical(
        kmp_Indent *loc,
        int32_t global_tid,
        kmp_CriticalName *lck)
{
    PRINT0(LD_IO, "call to kmpc_critical()\n");

    __MSPACE omptarget_hsail_TeamDescr *teamDescr = getMyTeamDescriptor();

    omp_set_lock(&teamDescr->criticalLock);
}

EXTERN void __kmpc_end_critical(
        kmp_Indent *loc,
        int32_t global_tid,
        kmp_CriticalName *lck)
{
    PRINT0(LD_IO, "call to kmpc_end_critical()\n");

    __MSPACE omptarget_hsail_TeamDescr * teamDescr = getMyTeamDescriptor();

    omp_unset_lock(&teamDescr->criticalLock);
}


