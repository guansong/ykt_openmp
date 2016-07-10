#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# Try to detect in the system several dependencies required by the different
# components of libomptarget. These are the dependencies we have:
#
# libelf : required by some targets to handle the ELF files at runtime.
# libffi : required to launch target kernels given function and argument 
#          pointers.
# CUDA : required to control offloading to NVIDIA GPUs.

include (FindPackageHandleStandardArgs)

find_package(PkgConfig)

################################################################################
# Looking for libelf...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBELF QUIET libelf)

find_path (
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR
  NAMES
    libelf.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    ENV CPATH
  PATH_SUFFIXES
    libelf)

find_library (
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES
  NAMES
    elf
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH)
    
set(LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS ${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBELF 
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS 
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES)
  
################################################################################
# Looking for libffi...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBFFI QUIET libffi)

find_path (
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR
  NAMES
    ffi.h
  HINTS
    ${LIBOMPTARGET_SEARCH_LIBFFI_INCLUDEDIR}
    ${LIBOMPTARGET_SEARCH_LIBFFI_INCLUDE_DIRS}
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    ENV CPATH)

# Don't bother look for the library if the header files were not found.
if (LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR)
  find_library (
      LIBOMPTARGET_DEP_LIBFFI_LIBRARIES
    NAMES
      ffi
    HINTS
      ${LIBOMPTARGET_SEARCH_LIBFFI_LIBDIR}
      ${LIBOMPTARGET_SEARCH_LIBFFI_LIBRARY_DIRS}
    PATHS
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
      ENV LIBRARY_PATH
      ENV LD_LIBRARY_PATH)
endif()

set(LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS ${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBFFI 
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBFFI_LIBRARIES
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS 
  LIBOMPTARGET_DEP_LIBFFI_LIBRARIES)
  
################################################################################
# Looking for CUDA...
################################################################################
find_package(CUDA QUIET)

set(LIBOMPTARGET_DEP_CUDA_FOUND ${CUDA_FOUND})
set(LIBOMPTARGET_DEP_CUDA_LIBRARIES ${CUDA_LIBRARIES})
set(LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

mark_as_advanced(
  LIBOMPTARGET_DEP_CUDA_FOUND 
  LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS
  LIBOMPTARGET_DEP_CUDA_LIBRARIES)


################################################################################
# Looking for ROCM...
################################################################################
pkg_check_modules(LIBOMPTARGET_SEARCH_LIBHSA QUIET libhsa-runtime64)

find_path (
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS
  NAMES
  hsa.h
  PATHS
  $ENV{HSA_RUNTIME_PATH}/include
  /opt/rocm/include/hsa
  /usr/local/include
  )

find_path (
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS
  NAMES
  libhsa-runtime64.so
  PATHS
  $ENV{HSA_RUNTIME_PATH}/lib
  /opt/rocm/lib/
  /usr/local/lib
  )

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBHSA
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS
  LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS)

# Do not use this until we are ready
if(0)
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    # require at least clang 3.9 for cl support
    if (NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 3.9)
      set(LIBOMPTARGET_DEVICE_COMPILER "clang")
      set(LIBOMPTARGET_USE_BUILD_COMPILER true)
    endif()
  endif()
endif()

if (CLC)
  find_path (
    LIBOMPTARGET_CLC_DIR
    NAMES
    clc2
    PATHS
    $ENV{HSA_CLC_PATH}
    /opt/rocm/hlc3.2/bin
    )

  if (LIBOMPTARGET_CLC_DIR)
    set(LIBOMPTARGET_DEVICE_COMPILER "clc2")
  endif()
endif()

if (NOT LIBOMPTARGET_DEVICE_COMPILER)
  find_path (
    LIBOMPTARGET_CLANG_DIR
    NAMES
    clang-3.9
    PATHS
    $ENV{CLANG_OCL_PATH}
    /opt/amd/llvm/bin
    )

  if (LIBOMPTARGET_CLANG_DIR)
    set(LIBOMPTARGET_DEVICE_COMPILER "clang")
    set(LIBOMPTARGET_DEVICE_COMPILER_INC_DIR ${LIBOMPTARGET_CLC_INC_DIR})
  endif()
endif()

