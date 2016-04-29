##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build the AMDGCN Device RTL if cloc.sh is available
#
##===----------------------------------------------------------------------===##
if(LIBOMPTARGET_AMDGCN_CLOC_COMPILER)
   find_program(amdgcn_cloc NAMES ${LIBOMPTARGET_AMDGCN_CLOC_COMPILER})
   if(amdgcn_cloc)
  
      libomptarget_say("Building AMDGCN device RTL. ")

      # This is a list of the cl files to compile with cloc
      set(amdgcn_components
         cancel
         critical
         debug
         Intrinsic
         libcall
         loop
         omptarget-hsail
         parallel
         reduction
         stdio
         sync
         task
      )

      # cmake caller should set LIBOMPTARGET_AMDGCN_MCPU to `mymcpu` 
      # but set to fiji otherwise.  Eventually, we will build ALL mcpus
      if(LIBOMPTARGET_AMDGCN_MCPU)
         set(amdgcn_mcpu ${LIBOMPTARGET_AMDGCN_MCPU})
      else()
         set(amdgcn_mcpu fiji)
      endif()
  
      foreach(cn ${amdgcn_components})
         set(amdgcn_bc_files ${amdgcn_bc_files} ${cn}.bc)
         add_custom_command(
            OUTPUT ${cn}.bc 
            # cloc.sh does not have a front-end only option, so we need to spec the entire command
            COMMAND /opt/amd/llvm/bin/clang -std=CL2.0 -include clc/clc.h -I/opt/rocm/libamdgcn/include -c -target amdgcn--amdhsa -D__OPENCL_VERSION__=200 -Dcl_khr_fp64 -Dcl_clang_storage_class_specifiers -Dcl_khr_int64 -emit-llvm -fno-builtin -mcpu=${amdgcn_mcpu} ${CMAKE_CURRENT_SOURCE_DIR}/src/${cn}.cl  -o ${cn}.bc
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${cn}.cl" 
         )
         add_custom_target(${cn} DEPENDS ${cn}.bc)
      endforeach()

      # Build the compsite bc file by linking the bc for each cl file
      add_custom_command(OUTPUT libomptarget-amdgcn-${amdgcn_mcpu}.bc COMMAND /opt/amd/llvm/bin/llvm-link ${amdgcn_bc_files} -o libomptarget-amdgcn-${amdgcn_mcpu}.bc DEPENDS ${amdgcn_components})
      add_custom_target(omptarget-amdgcn DEPENDS libomptarget-amdgcn-${amdgcn_mcpu}.bc)
      add_dependencies(omptarget omptarget-amdgcn)

      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-amdgcn-${amdgcn_mcpu}.bc DESTINATION "lib")

   else()
      libomptarget_say("Not building AMDGCN device RTL: cloc.sh not found")
   endif()
else()
   libomptarget_say("Not building AMDGCN device RTL: LIBOMPTARGET_AMDGCN_CLOC_COMPILER not set")
endif()
