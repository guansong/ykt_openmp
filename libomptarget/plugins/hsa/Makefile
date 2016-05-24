##===- RTLs/hsa/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build an RTL for a hsa machine if available
# github: gansong (zhang.guansong@gmail.com)
#
##===----------------------------------------------------------------------===##

#locate hsa libraries

SHELL := /bin/bash

hsalibs :=
ifneq ($(wildcard /opt/rocm/lib),)
  hsalibs += -L /opt/rocm/lib -lhsa-runtime64
endif
ifneq ($(wildcard $(HSA_RUNTIME_PATH)/lib),)
  hsalibs += -L $(HSA_RUNTIME_PATH)/lib -lhsa-runtime64
endif

hsaincludes :=
ifneq ($(wildcard /opt/rocm/include/hsa/hsa.h),)
  hsaincludes += -I /opt/rocm/hsa/include
endif
ifneq ($(wildcard $(HSA_RUNTIME_PATH)/include/hsa.h),)
  hsaincludes += -I $(HSA_RUNTIME_PATH)/include
endif
ifneq ($(wildcard $(HSA_RUNTIME_PATH)/include/hsa/hsa.h),)
  hsaincludes += -I $(HSA_RUNTIME_PATH)/include/hsa
endif

hsalib := $(findstring hsa, $(shell ldconfig -p | grep 'libhsa' | grep -o '/.*' ))

build_hsa :=
ifneq ($(hsalib),)
	build_hsa := true
endif

CPP_FILES = $(wildcard src/*.cpp)
INC_FILES = $(wildcard src/*.h)
OBJ_FILES = $(subst src/,obj/,$(CPP_FILES:.cpp=.o))

CC := g++
CFLAGS := -c -fPIC -I src/ -I../../src/ $(hsaincludes) -DTARGET_NAME=AMDHSA
LDFLAGS := -shared -lelf $(hsalibs)

ifdef OMPTARGET_DEBUG
CFLAGS += -g -DOMPTARGET_DEBUG -DHSA_ERROR_REPORT
endif

ifdef HSA10P
CFLAGS += -DHSA10P
endif

ifneq ($(build_hsa),)
../../lib/libomptarget.rtl.hsa.so : $(OBJ_FILES)
	@ echo "Building HSA target RTL" && \
	mkdir -p ../../lib
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES)
	
else
all : 
	@ echo "HSA not found in system: not building HSA RTL"
	
endif

obj/%.o: src/%.cpp $(INC_FILES) ../../deviceRTLs/amdgcn/src/option.h
	@ mkdir -p obj
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf obj

