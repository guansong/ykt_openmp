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
   if(LIBOMPTARGET_AMDGCN_ALTERNATE_HOST_COMPILER)
      find_program(AMDGCN_HOST_COMPILER NAMES ${LIBOMPTARGET_AMDGCN_ALTERNATE_HOST_COMPILER})
      if(AMDGCN_HOST_COMPILER)
         if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")

            libomptarget_say("Building HSA offloading plugin. ... ")

            # Define the suffix for the runtime messaging dumps.
            add_definitions(-DTARGET_NAME=HSA)
       
            include_directories(${LIBOMPTARGET_DEP_HSA_INCLUDE_DIRS})
    
            add_library(omptarget.rtl.hsa SHARED src/rtl.cpp)
      
            # Install plugin under the lib destination folder.
            install(TARGETS omptarget.rtl.hsa LIBRARY DESTINATION "lib")
      
            target_link_libraries(omptarget.rtl.hsa 
               ${LIBOMPTARGET_DEP_HSA_LIBRARIES} 
               hsa-runtime64 
               ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
               "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
            )
   
            # Report to the parent scope that we are building a plugin for hsa
            set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-hsa" PARENT_SCOPE)

         else()
            libomptarget_say("Not building HSA plugin: only support HSA in Linux x86_64 or ppc64le hosts.")
         endif()
      else()
         libomptarget_say("Not building HSA plugin: Invalid value for LIBOMPTARGET_AMDGCN_ALTERNATIVE_HOST_COMPILER")
      endif()
   else()
      libomptarget_say("Not building HSA plugin: No value given for LIBOMPTARGET_AMDGCN_ALTERNATIVE_HOST_COMPILER")
   endif()
else()
   libomptarget_say("Not building HSA plugin: LIBELF not found")
endif()
