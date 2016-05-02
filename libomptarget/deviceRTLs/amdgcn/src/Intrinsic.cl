//===--- intrinsic.cl - HSAIL OpenMP GPU initialization ---- HSA -*-===//
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

/*
   get_work_dim     Number of dimensions in use
   get_global_size  Number of global work items
   get_global_id    Global work item ID value
   get_local_size   Number of local work items
   get_local_id     Local work item ID
   get_num_groups   Number of work groups
   get_group_id     Work group ID
 */

EXTERN int __kmpc_ocl_get_global_size()
{
    return get_global_size(0);
}

EXTERN int __kmpc_ocl_get_global_id()
{
    return get_global_id(0);
}

EXTERN int __kmpc_ocl_get_local_size()
{
    return get_local_size(0);
}

EXTERN int __kmpc_ocl_get_local_id()
{
    return get_local_id(0);
}

EXTERN int __kmpc_ocl_get_num_groups()
{
    return get_num_groups(0);
}

EXTERN int __kmpc_ocl_get_group_id()
{
    return get_group_id(0);
}

EXTERN void __kmpc_ocl_barrier()
{
    /*
       CLK_LOCAL_MEM_FENCE - The barrier function will either
       flush any variables stored in local memory or
       queue a memory fence to ensure correct ordering of memory operations
       to local memory.

       CLK_GLOBAL_MEM_FENCE - The barrier function will queue a memory fence
       to ensure correct ordering of memory operations to global memory.
       This can be useful when work-items, for example, write to buffer or
       image objects and then want to read the updated data.

      */
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}

