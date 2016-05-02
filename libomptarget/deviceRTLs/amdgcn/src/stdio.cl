//===------------- stdio.cu - hsail OpenMP Std I/O --------------- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements standard IO functions. Note that varargs are not supported in
// CUDA, therefore the compiler needs to analyze the arguments passed to
// printf and generate a call to one of the functions defined here.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//


#include "omptarget-hsail.h"

EXTERN int __kmpc_printf(const char* str)
{
    PRINT0(LD_IO, "call to printf()\n");
    return 0;
}


