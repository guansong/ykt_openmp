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

# Found device compiler
if(LIBOMPTARGET_DEVICE_COMPILER)
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
    #set(amdgcn_mcpu fiji)
    set(amdgcn_mcpu hsail)
  endif()

  if(LIBOMPTARGET_CLC_DIR)
    set(amdgcn_mcpu hsail)
  endif()

  # Activate RTL message dumps if requested by the user.
  if(LIBOMPTARGET_AMDGCN_DEBUG)
    set(AMDGCN_DEBUG -DOMPTARGET_AMDGCN_DEBUG=-1 -g)
  endif()

  #MESSAGE( STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
  #MESSAGE( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )

  if(NOT LIBOMPTARGET_CLC_DIR)
    # special case when CC is set with ccache and compiler
    if (LIBOMPTARGET_USE_BUILD_COMPILER)
      # FIXME: How to set DL in this path
      if (CMAKE_C_COMPILER MATCHES "ccache")
        #MESSAGE( STATUS "CMAKE_C_COMPILER_ARG1: " ${CMAKE_C_COMPILER_ARG1} )
        STRING(REGEX REPLACE " " "" CMAKE_C_COMPILER_ARG1 ${CMAKE_C_COMPILER_ARG1})
        #MESSAGE( STATUS "CMAKE_C_COMPILER_ARG1: " ${CMAKE_C_COMPILER_ARG1} )
        set(DC ${CMAKE_C_COMPILER_ARG1})
      else()
        set(DC ${CMAKE_C_COMPILER})
      endif()
    else ()
      set(DC ${LIBOMPTARGET_CLANG_DIR}/clang)
      set(DL ${LIBOMPTARGET_CLANG_DIR}/llvm-link)
    endif()
  else()
    set(DC ${LIBOMPTARGET_CLC_DIR}/clc2)
    set(DL ${LIBOMPTARGET_CLC_DIR}/llvm-link)
  endif()

  if(NOT LIBOMPTARGET_CLC_DIR)
    set(CLANGFLAGS -Xclang -cl-std=CL2.0)
    set(CLANGFLAGS ${CLANGFLAGS} -D__OPENCL_VERSION__=200)

    #set(CLANGFLAGS ${CLANGFLAGS} -I${LIBOMPTARGET_CLC_INC_DIR} -include clc/clc.h)
    #set(CLANGFLAGS ${CLANGFLAGS} -Dcl_clang_storage_class_specifiers)

    #set(CLANGFLAGS ${CLANGFLAGS} -Dcl_khr_int64)
    #set(CLANGFLAGS ${CLANGFLAGS} -Dcl_khr_fp64)

    # May need to turn this one back on?
    #set(CLANGFLAGS ${CLANGFLAGS} -fno-builtin)

    set(CLANGFLAGS ${CLANGFLAGS} -target spir64-unknown-unknown)

    # emit-llvm for cpu???
    if (NOT amdgcn_mcpu MATCHES "hsail")
      set(CLANGFLAGS ${CLANGFLAGS} -mcpu=${amdgcn_mcpu})
    endif()

    set(CLANGFLAGS ${CLANGFLAGS} -emit-llvm -c)

    set(DCFLAGS ${CLANGFLAGS})
  else()
    set(CLCFLAGS -cl-std=CL2.0)
    set(CLCFLAGS ${CLCFLAGS} -D_WITHOUT_STDDEF_HEADER_)

    set(DCFLAGS ${CLCFLAGS})
  endif()

  #MESSAGE( STATUS "Device COMPILER: ${DC}" )
  #MESSAGE( STATUS "Device COMPILER FLAGS: ${DCFLAGS}" )
  libomptarget_say("Device COMPILER set as ${DC}")

  #if (CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  #  libomptarget_say("Archive dir: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
  #endif()

  foreach(cn ${amdgcn_components})
    set(amdgcn_bc_files ${amdgcn_bc_files} ${cn}.bc)
    add_custom_command(
      OUTPUT ${cn}.bc
      COMMAND ${DC} ${DCFLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/src/${cn}.cl -o ${cn}.bc
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${cn}.cl" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
      )
    add_custom_target(${cn} DEPENDS ${cn}.bc)
  endforeach()

  # Build the compsite bc file by linking the bc for each cl file

  if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    add_custom_command(
      OUTPUT libomptarget-amdgcn-${amdgcn_mcpu}.bc
      COMMAND ${DL} ${amdgcn_bc_files} -o libomptarget-amdgcn-${amdgcn_mcpu}.bc
      DEPENDS ${amdgcn_bc_files}
      )
  else()
    add_custom_command(
      OUTPUT libomptarget-amdgcn-${amdgcn_mcpu}.bc
      COMMAND ${DL} ${amdgcn_bc_files} -o ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/libomptarget-amdgcn-${amdgcn_mcpu}.bc
      DEPENDS ${amdgcn_bc_files}
      )
  endif()

  add_custom_target(omptarget-amdgcn DEPENDS libomptarget-amdgcn-${amdgcn_mcpu}.bc)
  add_dependencies(omptarget omptarget-amdgcn)

  if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-amdgcn-${amdgcn_mcpu}.bc DESTINATION "lib")
  else()
    install(FILES ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/libomptarget-amdgcn-${amdgcn_mcpu}.bc DESTINATION "lib")
  endif()

else()
  libomptarget_say("Not building AMDGCN device RTL: Device compiler not found")
endif()
