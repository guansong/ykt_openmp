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

INC_FILES = $(wildcard *.h)
CPP_FILES = $(wildcard *.cpp)
OCL_FILES = $(wildcard *.cl)

CXX := g++
CXX_FLAGS := -I /opt/AMDAPP/include

LNK_FLAGS := -L /opt/AMDAPP/lib/x86_64/ -lOpenCL

#this is only the current dir, not the makefile dir
CURDIR := $(shell pwd)

CLC2_FLAGS := -w -cl-std=CL2.0 -D_WITHOUT_STDDEF_HEADER_ -I$(CURDIR) -DOMPTARGET_HSAIL_DEBUG=-1
#CLC_FLAGS := -save-temps-all -cl-std=CL1.2 -D_WITHOUT_STDDEF_HEADER_ -I$(CURDIR) -DOMPTARGET_HSAIL_DEBUG=-1
#CLC_FLAGS := -save-temps-all -cl-std=CL2.0 -D_WITHOUT_STDDEF_HEADER_ -I$(CURDIR) -DOMPTARGET_HSAIL_DEBUG=-1 -g -O0
#CLC_FLAGS := -cl-std=CL2.0 -D_WITHOUT_STDDEF_HEADER_ -I$(CURDIR)"

CLC1_FLAGS := -w -cl-std=CL1.2 -D_WITHOUT_STDDEF_HEADER_ -I$(CURDIR) -DOMPTARGET_HSAIL_DEBUG=-1

AOC := aoc2
AOC_FLAGS := -cl-std=CL2.0 -D_WITHOUT_STDDEF_HEADER_ -I$(CURDIR) -DOMPTARGET_HSAIL_DEBUG=-1
#AOC_FLAGS += -save-temps-all
# arch and device
AOC_FLAGS += -march=hsail-64 -mdevice=Spectre

CLANG := clang
CLANGFLAGS := -Xclang -cl-std=CL2.0 -D__OPENCL_VERSION__=200
CLANGFLAGS += -Wno-typedef-redefinition
CLANGFLAGS += -Xclang -include -Xclang $(BITCODE_HEADER) -Dcl_clang_storage_class_specifiers
# using C calling convention
CLANGFLAGS += -target amdgcn--amdhsa
# using spir calling calling convention
#CLANGFLAGS += -target spir64-unknown-unknown
CLANGFLAGS += -emit-llvm -c

LNK := llvm-link
LNKFLAGS := -suppress-warnings
LNKLIBS := ${HSA_BUILTIN_PATH}/hsa_math.bc
LNKLIBS += ${HSA_BUILTIN_PATH}/builtins-hsail.opt.bc
LNKLIBS += ${HSA_BUILTIN_PATH}/hsail-amdgpu-wrapper.ll

LLC := llc
LLCFLAGS := -mtriple amdgcn--amdhsa
LLCFLAGS += -mcpu=kaveri
#LLCFLAGS += -filetype=asm
LLCFLAGS += -filetype=obj

LLD := lld
LLDFLAGS := -flavor gnu -shared


#llc -mtriple amdgcn--amdhsa -mcpu=kaveri -filetype=asm -o linked.s hello_world.bc
#llc -mtriple amdgcn--amdhsa -mcpu=kaveri -filetype=obj -o linked.o hello_world.bc
#lld -flavor gnu -shared -o a.out linked.o

all: driver test

compile:
	@echo
	@echo ------------------------- compile -------------------------
	$(CLANG) $(CLANGFLAGS) driver.cl
	$(LNK) $(LNKFLAGS) -o driver.lnk.bc driver.bc ${LNKLIBS}
	$(LLC) $(LLCFLAGS) -o kernel.o driver.lnk.bc
	$(LLD) $(LLDFLAGS) -o kernel kernel.o


test: driver
	@echo
	@echo ------------------------- test 2.0 b -----------------------
	$< -b
	@echo ------------------------- test 2.0 s -----------------------
	env AMD_OCL_BUILD_OPTIONS_APPEND="$(CLC2_FLAGS)" $< -s

#driver: driver.o
driver: driver.bin driver.o
	$(CXX) driver.o $(LNK_FLAGS) -o $@

%.bin: %.cl ../src/*.h ../src/*.cl $(INC_FILES)
	$(AOC) $(AOC_FLAGS) $<

%.o: %.cpp $(INC_FILES)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

clean:
	rm -rf _temp* *.o driver.bin driver

