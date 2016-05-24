##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build a plugin for an HSA machine if available.
#
##===----------------------------------------------------------------------===##

if(LIBOMPTARGET_DEP_LIBELF_FOUND)
   if(LIBOMPTARGET_AMDGCN_CLOC_COMPILER)
      find_program(amdgcn_cloc NAMES ${LIBOMPTARGET_AMDGCN_CLOC_COMPILER})
      if(amdgcn_cloc)
         if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")

            libomptarget_say("Building HSA offloading plugin. ... ")

            find_path (
              LIBOMPTARGET_DEP_HSA_INCLUDE_DIRS
              NAMES
              hsa.h
              PATHS
              $ENV{HSA_RUNTIME_PATH}/include
              /opt/rocm/include/hsa
              /usr/local/include
              )

            find_path (
              LIBOMPTARGET_DEP_HSA_LIBRARIES_DIRS
              NAMES
              libhsa-runtime64.so
              PATHS
              $ENV{HSA_RUNTIME_PATH}/lib
              /opt/rocm/lib/
              /usr/local/lib
              )

            # Define the suffix for the runtime messaging dumps.
            add_definitions(-DTARGET_NAME=HSA)
       
            include_directories(${LIBOMPTARGET_DEP_HSA_INCLUDE_DIRS})
    
            add_library(omptarget.rtl.hsa SHARED src/rtl.cpp src/elf_utils.cpp)
      
            # Install plugin under the lib destination folder.
            install(TARGETS omptarget.rtl.hsa LIBRARY DESTINATION "lib")
      
            target_link_libraries(
               omptarget.rtl.hsa 
                  -L${LIBOMPTARGET_DEP_HSA_LIBRARIES_DIRS} -lhsa-runtime64
                  ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
                  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
            )
   
            # Report to the parent scope that we are building a plugin for hsa
            set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-hsa" PARENT_SCOPE)

         else()
            libomptarget_say("Not building HSA plugin: only support HSA in Linux x86_64 or ppc64le hosts.")
         endif()
      else()
         libomptarget_say("Not building HSA plugin: Invalid value for LIBOMPTARGET_AMDGCN_CLOC_COMPILER")
      endif()
   else()
      libomptarget_say("Not building HSA plugin: No value given for LIBOMPTARGET_AMDGCN_CLOC_COMPILER")
   endif()
else()
   libomptarget_say("Not building HSA plugin: LIBELF not found")
endif()
