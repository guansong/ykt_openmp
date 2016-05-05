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
# bitbucket: gansong (zhang.guansong@gmail.com)
#
##===----------------------------------------------------------------------===##

#locate hsa libraries

SHELL := /bin/bash

hsalib := $(findstring hsa, $(shell ldconfig -p | grep 'libhsa' | grep -o '/.*' ))

build_hsa :=
ifneq ($(hsalib),)
	build_hsa := true
endif

CPP_FILES = $(wildcard src/*.cpp)
CL_FILES = $(wildcard src/*.cl)
LL_FILES = $(wildcard src/*.ll)

INC_FILES = $(wildcard src/*.h)

CLBC_FILES = $(subst src/,obj/,$(CL_FILES:.cl=.bc))
LLBC_FILES = $(subst src/,obj/,$(LL_FILES:.ll=.bc))

ifneq ($(wildcard /opt/rocm/hlc3.2/bin/clc2),)
CLC := /opt/rocm/hlc3.2/bin/clc2
endif
ifeq ($(CLC),)
CLC := clc2
endif

CLCFLAGS := -cl-std=CL2.0 -D_WITHOUT_STDDEF_HEADER_

AOC := aoc2
AOCFLAGS := -march=hsail-64 -mdevice=Spectre -save-temps-all -cl-std=CL2.0 -D_WITHOUT_STDDEF_HEADER_
AOCFLAGS += -I./src

# clang on top of the search path
CLANG := clang
# clang basic flag
CLANGFLAGS := -Xclang -cl-std=CL2.0 -D__OPENCL_VERSION__=200
CLANGFLAGS += -Wno-typedef-redefinition
# include libclc header
CLANGFLAGS += -Xclang -include -Xclang $(BITCODE_HEADER) -Dcl_clang_storage_class_specifiers
# using C calling convention
CLANGFLAGS += -target amdgcn--amdhsa
# using spir calling calling convention
#CLANGFLAGS += -target spir64-unknown-unknown
# generat bc files
CLANGFLAGS += -emit-llvm -c

# make sure we use the same llvm from clang
LLVM-AS := `which clang | sed 's/clang$$/llvm-as/'`
LLVM-DIS := llvm-dis

ifneq ($(wildcard /opt/rocm/hcc-hsail/hlc/bin/llvm-link),)
BCLNK := /opt/rocm/hcc-hsail/hlc/bin/llvm-link
endif
ifneq ($(wildcard $(HSA_HLC_PATH)/llvm-link),)
BCLNK := $(HSA_HLC_PATH)/llvm-link
endif

LNKFLAGS :=
#-suppress-warnings

ifdef OMPTARGET_HSAIL_DEBUG
CLCFLAGS += -DOMPTARGET_HSAIL_DEBUG=-1 -g -DHSA_ERROR_REPORT
endif

ifneq ($(build_hsa),)
../../lib/libomptarget-hsail.bc : $(CLBC_FILES) $(LLBC_FILES)
	@ echo "Building HSA target RTL" && \
	mkdir -p ../../lib
	$(BCLNK) $(LNKFLAGS) -o $@ $(CLBC_FILES) $(LLBC_FILES)

else
all :
	@ echo "HSA not found in system: not building HSA RTL"

endif

obj/%.bc: src/%.cl $(INC_FILES)
	@ mkdir -p obj
	#$(CLANG) $(CLANGFLAGS) $< -o obj/$*.bc
	$(CLC) $(CLCFLAGS) $< -o obj/$*.bc
	@#$(AOC) $(AOCFLAGS) $< && mv _temp_0_Spectre_original.bc obj/$*.bc && rm _temp_*

obj/%.bc: src/%.ll
	@ mkdir -p obj
	$(LLVM-AS) $< -o obj/$*.bc

clean:
	rm -rf obj

