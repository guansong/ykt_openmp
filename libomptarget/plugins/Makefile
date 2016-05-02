##===- RTLs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build an RTL for this machine is available
#
##===----------------------------------------------------------------------===##

#RTL_FOLDERS = $(wildcard */)
PLUGIN_FOLDERS = cuda hsa ppc64 ppc64le x86_64


all:
	@for f in $(PLUGIN_FOLDERS); do \
		if test -d $$f; then make -C $$f; fi; \
	done 
	
clean:
	@for f in $(PLUGIN_FOLDERS); do \
		if test -d $$f; then make -C $$f clean; fi; \
	done 
