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

if(LIBOMPTARGET_AMDGCN_ALTERNATE_HOST_COMPILER)
  find_program(AMDGCN_HOST_COMPILER NAMES ${LIBOMPTARGET_AMDGCN_ALTERNATE_HOST_COMPILER})
  if(NOT AMDGCN_HOST_COMPILER)
    libomptarget_say("Not building AMDGCN device RTL: invalid AMDGCN alternate host compiler.")
  endif()
  
  libomptarget_say("Building AMDGCN device RTL. ")

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

  # cmake caller should set LIBOMPTARGET_AMDGCN_COMPUTE_CAPABILITY=`mymcpu` to build for current processor 
  # but set to fiji otherwise.
  if(LIBOMPTARGET_AMDGCN_COMPUTE_CAPABILITY)
    set(AMDGCN_ARCH ${LIBOMPTARGET_AMDGCN_COMPUTE_CAPABILITY})
  else()
    set(AMDGCN_ARCH fiji)
  endif()
  
  # Activate RTL message dumps if requested by the user.
  if(LIBOMPTARGET_AMDGCN_DEBUG)
    set(AMDGCN_DEBUG -DOMPTARGET_AMDGCN_DEBUG=-1 -g)
  endif()

  foreach(cn ${amdgcn_components})
     set(amdgcn_bc_files ${amdgcn_bc_files} ${cn}.bc)
     add_custom_command(
        OUTPUT ${cn}.bc 
        COMMAND /opt/amd/llvm/bin/clang -std=CL2.0 -include clc/clc.h -I/opt/rocm/libamdgcn/include -c -target amdgcn--amdhsa -D__OPENCL_VERSION__=200 -Dcl_khr_fp64 -Dcl_clang_storage_class_specifiers -Dcl_khr_int64 -emit-llvm -fno-builtin -mcpu=${AMDGCN_ARCH} ${CMAKE_CURRENT_SOURCE_DIR}/src/${cn}.cl  -o ${cn}.bc
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${cn}.cl" 
     )
     add_custom_target(${cn} DEPENDS ${cn}.bc)
  endforeach()
  add_custom_command(OUTPUT omptarget-amdgcn-${AMDGCN_ARCH}.bc COMMAND /opt/amd/llvm/bin/llvm-link ${amdgcn_bc_files} -o omptarget-amdgcn-${AMDGCN_ARCH}.bc DEPENDS ${amdgcn_components})
  add_custom_target(omptarget-amdgcn DEPENDS omptarget-amdgcn-${AMDGCN_ARCH}.bc)
  add_dependencies(omptarget omptarget-amdgcn)

  install(FILES omptarget-amdgcn.bc DESTINATION "lib")

  #target_link_libraries(omptarget-nvptx ${AMDGCN_LIBRARIES})

else()
  libomptarget_say("Not building AMDGCN device RTL: LIBOMPTARGET_AMDGCN_ALTERNATIVE_HOST_COMPILER not set")
endif()
