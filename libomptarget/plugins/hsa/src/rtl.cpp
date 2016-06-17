//===----RTLs/hsa/src/rtl.cpp - Target RTLs Implementation -------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// RTL for hsa machine
// github: guansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <dlfcn.h>
#include <elf.h>
#include <ffi.h>
#include <gelf.h>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "elf_utils.h"

#include "omptarget.h"

/* Sync the limits with device, as we use static structure */
#include "../../../deviceRTLs/amdgcn/src/option.h"

#ifndef TARGET_NAME
#define TARGET_NAME AMDHSA
#endif

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)
#define DP(...) DEBUGP("Target " GETNAME(TARGET_NAME) " RTL",__VA_ARGS__)

#ifdef OMPTARGET_DEBUG
#define check(msg, status) \
  if (status != HSA_STATUS_SUCCESS) { \
    /* fprintf(stderr, "[%s:%d] %s failed.\n", __FILE__, __LINE__, #msg);*/ \
    DP(#msg" failed\n") \
    /*assert(0);*/ \
  } else { \
    /* fprintf(stderr, "[%s:%d] %s succeeded.\n", __FILE__, __LINE__, #msg); */ \
    DP(#msg" succeeded\n") \
  }
#else
#define check(msg, status) \
{}
#endif

extern MODULE_t __hsaBrigModule;

/*
 * Define required BRIG data structures.
 */

typedef uint32_t BrigCodeOffset32_t;

typedef uint32_t BrigDataOffset32_t;

typedef uint16_t BrigKinds16_t;

typedef uint8_t BrigLinkage8_t;

typedef uint8_t BrigExecutableModifier8_t;

typedef BrigDataOffset32_t BrigDataOffsetString32_t;

enum BrigKinds {
  BRIG_KIND_NONE = 0x0000,
  BRIG_KIND_DIRECTIVE_BEGIN = 0x1000,
  BRIG_KIND_DIRECTIVE_KERNEL = 0x1008,
};

typedef struct BrigBase BrigBase;
struct BrigBase {
  uint16_t byteCount;
  BrigKinds16_t kind;
};

typedef struct BrigExecutableModifier BrigExecutableModifier;
struct BrigExecutableModifier {
  BrigExecutableModifier8_t allBits;
};

typedef struct BrigDirectiveExecutable BrigDirectiveExecutable;
struct BrigDirectiveExecutable {
  uint16_t byteCount;
  BrigKinds16_t kind;
  BrigDataOffsetString32_t name;
  uint16_t outArgCount;
  uint16_t inArgCount;
  BrigCodeOffset32_t firstInArg;
  BrigCodeOffset32_t firstCodeBlockEntry;
  BrigCodeOffset32_t nextModuleEntry;
  uint32_t codeBlockEntryCount;
  BrigExecutableModifier modifier;
  BrigLinkage8_t linkage;
  uint16_t reserved;
};

typedef struct BrigData BrigData;
struct BrigData {
  uint32_t byteCount;
  uint8_t bytes[1];
};

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
static hsa_status_t find_gpu(hsa_agent_t agent, void *data)
{
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  hsa_device_type_t device_type;
  hsa_status_t stat =
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }
  if (device_type == HSA_DEVICE_TYPE_GPU) {
    *((hsa_agent_t *)data) = agent;
  }
  return HSA_STATUS_SUCCESS;
}

/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void* data) {
  hsa_region_segment_t segment;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (HSA_REGION_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    hsa_region_t* ret = (hsa_region_t*) data;
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
static hsa_status_t get_gpu_agent(hsa_agent_t agent, void *data) {
  hsa_status_t status;
  hsa_device_type_t device_type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_GPU == device_type) {
    hsa_agent_t* ret = (hsa_agent_t*)data;
    *ret = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}


/// Account the memory allocated per device
struct AllocMemEntryTy {
  unsigned TotalSize;
  std::vector<void*> Ptrs;

  AllocMemEntryTy() : TotalSize(0) {}
};

/// Keep entries table per device
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

/// Class containing all the device information
class RTLDeviceInfoTy {
  std::vector<FuncOrGblEntryTy> FuncGblEntries;

  std::vector<AllocMemEntryTy> MemoryEntries;

public:
  int NumberOfDevices;

  hsa_ext_program_t hsaProgram;
  MODULE_t* brig_module;

  //std::list<MODULE_t> hsaBrigModules;

  std::vector<int> ThreadsPerGroup;
  std::vector<int> GroupsPerDevice;

  hsa_agent_t agent;

  hsa_queue_t* commandQueue;

  // Record memory pointer associated with device
  void addMemory(int32_t device_id, void *ptr, int size){
    assert( device_id < MemoryEntries.size() && "Unexpected device id!");
    AllocMemEntryTy &E = MemoryEntries[device_id];
    E.TotalSize += size;
    E.Ptrs.push_back(ptr);
  }

  // Return true if the pointer is associated with device
  bool findMemory(int32_t device_id, void *ptr){
    assert( device_id < MemoryEntries.size() && "Unexpected device id!");
    AllocMemEntryTy &E = MemoryEntries[device_id];

    return  std::find(E.Ptrs.begin(),E.Ptrs.end(),ptr) != E.Ptrs.end();
  }

  // Remove pointer from the dtaa memory map
  void deleteMemory(int32_t device_id, void *ptr){
    assert( device_id < MemoryEntries.size() && "Unexpected device id!");
    AllocMemEntryTy &E = MemoryEntries[device_id];

    std::vector<void*>::iterator it = std::find(E.Ptrs.begin(),E.Ptrs.end(),ptr);

    if(it != E.Ptrs.end()){
      E.Ptrs.erase(it);
    }
  }

  // Record entry point associated with device
  void addOffloadEntry(int32_t device_id, __tgt_offload_entry entry ){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];

    E.Entries.push_back(entry);
  }

  // Return true if the entry is associated with device
  bool findOffloadEntry(int32_t device_id, void *addr){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];

    for(unsigned i=0; i<E.Entries.size(); ++i){
      if(E.Entries[i].addr == addr)
        return true;
    }

    return false;
  }

  // Return the pointer to the target entries table
  __tgt_target_table *getOffloadEntriesTable(int device_id){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];

    unsigned size = E.Entries.size();

    // Table is empty
    if(!size)
      return 0;

    __tgt_offload_entry *begin = &E.Entries[0];
    __tgt_offload_entry *end = &E.Entries[size-1];

    // Update table info according to the entries and return the pointer
    E.Table.EntriesBegin = begin;
    E.Table.EntriesEnd = ++end;

    return &E.Table;
  }

  // Clear entries table for a device
  void clearOffloadEntriesTable(int device_id){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];
    E.Entries.clear();
    E.Table.EntriesBegin = E.Table.EntriesEnd = 0;
  }

  RTLDeviceInfoTy(int num_devices) : brig_module(NULL), commandQueue(NULL) {
    // Code from the HSA example

    hsa_status_t err;

    /*
     * Initialize hsa runtime
     */
    err = hsa_init();
    check(Initializing the hsa runtime, err);

    /*
     * Iterate over the agents and pick the gpu agent using
     * the find_gpu callback.
     */
    err = hsa_iterate_agents(get_gpu_agent, &agent);
    if(err == HSA_STATUS_INFO_BREAK) { err = HSA_STATUS_SUCCESS; }
    check(Getting a gpu agent, err);

    /*
     * Query the name of the device.
     */
    char name[64] = { 0 };
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    check(Querying the agent name, err);
    //printf("The agent name is %s.\n", name);

    /*
     * Query the maximum size of the queue.
     */
    uint32_t queue_size = 0;
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    check(Querying the agent maximum queue size, err);
    //printf("The maximum queue size is %u.\n", (unsigned int) queue_size);

    /*
     * Create a queue using the maximum size.
     */
    err = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &commandQueue);
    check(Creating the queue, err);

    /*
     * Create hsa program.
     */
    memset(&hsaProgram,0,sizeof(hsa_ext_program_t));
    err = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &hsaProgram);
    check(Creating the hsa program, err);

    /*
     * hsa device number
     */
    if (num_devices) {
      NumberOfDevices=num_devices;
    }
    else {
      // FIXME: compute from hsa runtime
    }

    // All the hsa device have the same setting for wavefront and workgroup size for now
    /*
     * Query the wavefront size.
     */
    uint32_t wavefront_size = 0;
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size);
    check(Querying the agent maximum queue size, err);
    //printf("The wavefront size is %u.\n", (unsigned int) wavefront_size);

    /*
     * Query the workgroup size.
     */
    uint32_t workgroup_size = 0;
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &workgroup_size);
    check(Querying the agent maximum queue size, err);
    //printf("The max workgroup size is %u.\n", (unsigned int) workgroup_size);


    // Init the device info
    MemoryEntries.resize(NumberOfDevices);
    FuncGblEntries.resize(NumberOfDevices);

    ThreadsPerGroup.resize(NumberOfDevices);
    GroupsPerDevice.resize(NumberOfDevices);

    //Only one device here;
    for (int i =0; i<NumberOfDevices; i++) {
      ThreadsPerGroup[i]=workgroup_size;
      GroupsPerDevice[i]=0;

      DP("Device %d: queried limit for groupsPerDevice %d & threadsPerGroup %d\n",
          i, GroupsPerDevice[i], ThreadsPerGroup[i]);
    }
  }

  ~RTLDeviceInfoTy(){
    hsa_status_t err;

    DP("Destroying the DeviceInfo.\n");

    /*
     * Destroy hsa program
     */
    err=hsa_ext_program_destroy(hsaProgram);
    check(Destroying the program, err);

    /*
     * Destroy brig module
     */
    if (brig_module) {
      destroy_brig_module(brig_module);
      DP("Destroying brig module succeeded.\n");
    }

    /*
     * Destroy more things
     */
    //FIXME: isa, code_object, executable.
    //hsa_executable_create

    /*
     * Destroy the queue
     */
    if (commandQueue) {
      err=hsa_queue_destroy(commandQueue);
      check(Destroying the queue, err);
    }

    /*
     * Shutdown hsa runtime
     */
    err=hsa_shut_down();
    check(Shutting down the runtime, err);

    // Free devices allocated memory
    for(unsigned i=0; i<MemoryEntries.size(); ++i ) {
      for(unsigned j=0; j<MemoryEntries[i].Ptrs.size(); ++j ) {
        if(MemoryEntries[i].Ptrs[j]) {
          DP("Free memory at this address.\n");
          free(MemoryEntries[i].Ptrs[j]);
        }
      }
    }

    // Free devices allocated entry
    for(unsigned i=0; i<FuncGblEntries.size(); ++i ) {
      for(unsigned j=0; j<FuncGblEntries[i].Entries.size(); ++j ) {
        if(FuncGblEntries[i].Entries[j].addr) {
          DP("Free address %016llx.\n",(long long unsigned)(Elf64_Addr)FuncGblEntries[i].Entries[j].addr);
          free(FuncGblEntries[i].Entries[j].addr);
        }
      }
    }

    /*
    // we did not do anything with hsaBrigModules.
    for(std::list<MODULE_t>::iterator
      ii = hsaBrigModules.begin(), ie = hsaBrigModules.begin(); ii!=ie; ++ii) {
    }
    */
  }
};

// assume this is one for now;
// HSA runtime 1.0 seems only define one device for each type
#define NUMBER_OF_DEVICES 1

static RTLDeviceInfoTy DeviceInfo(NUMBER_OF_DEVICES);

#ifdef __cplusplus
extern "C" {
#endif

int __tgt_rtl_device_type(int device_id){

  if( device_id < DeviceInfo.NumberOfDevices)
    return 0; // HSA for now

  return -1;
}

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {

  // Is the library version incompatible with the header file?
  if (elf_version(EV_CURRENT) == EV_NONE) {
    DP("Incompatible ELF library!\n");
    return 0;
  }

  char *img_begin = (char *)image->ImageStart;
  char *img_end = (char *)image->ImageEnd;
  size_t img_size = img_end - img_begin;

  // Obtain elf handler
  Elf *e = elf_memory(img_begin, img_size);
  if (!e) {
    DP("Unable to get ELF handle: %s!\n", elf_errmsg(-1));
    return 0;
  }

  // Check if ELF is the right kind.
  if (elf_kind(e) != ELF_K_ELF) {
    DP("Unexpected ELF type!\n");
    return 0;
  }
  Elf64_Ehdr *eh64 = elf64_getehdr(e);
  Elf32_Ehdr *eh32 = elf32_getehdr(e);

  if (!eh64 && !eh32) {
    DP("Unable to get machine ID from ELF file!\n");
    elf_end(e);
    return 0;
  }

  uint16_t MachineID;
  if (eh64 && !eh32)
    MachineID = eh64->e_machine;
  else if (eh32 && !eh64)
    MachineID = eh32->e_machine;
  else {
    DP("Ambiguous ELF header!\n");
    elf_end(e);
    return 0;
  }

  elf_end(e);

  switch(MachineID) {
    // old brig file in HSA 1.0P
    case 0:
    // brig file in HSAIL path
    case 44890:
    case 44891:
      break;
    // amdgcn
    case 224:
      break;
    default:
      DP("Unsupported machine ID found: %d\n", MachineID);
      return 0;
  }

  return 1;
}

int __tgt_rtl_number_of_devices(){
  return DeviceInfo.NumberOfDevices;
}

int32_t __tgt_rtl_init_device(int device_id){
  // this is per device id init
  DP("Initialize the device id: %d\n", device_id);

  //DP("Initialize the device, GroupsPerDevice limit: %d\n", TEAMS_ABSOLUTE_LIMIT);
  //DP("Initialize the device, ThreadsPerGroup limit: %d\n", MAX_NUM_OMP_THREADS);

  // get the global one from device side as we use static structure
  int GroupsLimit = TEAMS_ABSOLUTE_LIMIT;
  int ThreadsLimit = MAX_NUM_OMP_THREADS/TEAMS_ABSOLUTE_LIMIT;

  if ((DeviceInfo.GroupsPerDevice[device_id] > GroupsLimit) ||
      (DeviceInfo.GroupsPerDevice[device_id] == 0)) {
    DeviceInfo.GroupsPerDevice[device_id]=GroupsLimit;
  }

  if ((DeviceInfo.ThreadsPerGroup[device_id] > ThreadsLimit) ||
      DeviceInfo.ThreadsPerGroup[device_id] == 0) {
    DeviceInfo.ThreadsPerGroup[device_id]=ThreadsLimit;
  }

  DP("Device %d: default limit for groupsPerDevice %d & threadsPerGroup %d\n",
      device_id,
      DeviceInfo.GroupsPerDevice[device_id],
      DeviceInfo.ThreadsPerGroup[device_id]);

  DP("Device %d: total threads %d x %d = %d\n",
      device_id,
      DeviceInfo.ThreadsPerGroup[device_id],
      DeviceInfo.GroupsPerDevice[device_id],
      DeviceInfo.GroupsPerDevice[device_id]*DeviceInfo.ThreadsPerGroup[device_id]);

  return OFFLOAD_SUCCESS ;
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id, __tgt_device_image *image){
  size_t img_size = (char*) image->ImageEnd - (char*) image->ImageStart;

  DeviceInfo.clearOffloadEntriesTable(device_id);

  int useBrig = 0;

  // We do not need to set the ELF version because the caller of this function
  // had to do that to decide the right runtime to use

  // Obtain elf handler and do an extra check
  {
    Elf *elfP = elf_memory ((char*)image->ImageStart, img_size);
    if(!elfP){
      DP("Unable to get ELF handle: %s!\n", elf_errmsg(-1));
      return 0;
    }

    if( elf_kind(elfP) !=  ELF_K_ELF){
      DP("Invalid Elf kind!\n");
      elf_end(elfP);
      return 0;
    }

    uint16_t MachineID;
    {
      Elf64_Ehdr *eh64 = elf64_getehdr(elfP);
      Elf32_Ehdr *eh32 = elf32_getehdr(elfP);
      if (eh64 && !eh32)
        MachineID = eh64->e_machine;
      else if (eh32 && !eh64)
        MachineID = eh32->e_machine;
      else{
        printf("Ambiguous ELF header!\n");
        return 0;
      }
    }

    switch(MachineID) {
      // old brig file in HSA 1.0P
      case 0:
      // brig file in HSAIL path
      case 44890:
      case 44891:
        {
          useBrig = 1;
        };
        break;
      case 224:
        // do nothing, amdgcn
        break;
      default:
        DP("Unsupported machine ID found: %d\n", MachineID);
        elf_end(elfP);
        return 0;
    }

    DP("Machine ID found: %d\n", MachineID);
    // Close elf
    elf_end(elfP);
  }

  // HSA runtime
  hsa_status_t err;

  // Create a BRIG module
  if (useBrig) {
    if (STATUS_SUCCESS == create_brig_module_from_brig_memory(
          (char *)image->ImageStart, img_size, &DeviceInfo.brig_module)) {
      DP("Creating brig module succeeded.\n");
    }
    else {
      DP("Creating brig module failed.\n");
      return 0;
    }
    //DP("brig_moduel %016llx.\n", (long long unsigned)(Elf64_Addr)(DeviceInfo.brig_module));
    //DP("__hsaBrigModule %016llx.\n", (long long unsigned)(Elf64_Addr)__hsaBrigModule);
  }

  /*
   * Add the BRIG module to hsa program.
   */
  if (useBrig) {
    // Crazy pointer casting!
    err = hsa_ext_program_add_module(DeviceInfo.hsaProgram, (MODULE_t)DeviceInfo.brig_module);
    //err = hsa_ext_program_add_module(DeviceInfo.hsaProgram, __hsaBrigModule);
    check(Adding the brig module to the program, err);
  }

  // FIXME: live range???
  // this is the primary structure to obtain
  hsa_executable_t executable;
  // this is the secondary structure
  hsa_code_object_t code_object;

  if (useBrig) {
    /*
     * Determine the agents ISA.
     */
    hsa_isa_t isa;
    err = hsa_agent_get_info(DeviceInfo.agent, HSA_AGENT_INFO_ISA, &isa);
    check(Query the agents isa, err);

    /*
     * Finalize the program and extract the code object.
     */
    // FIXME: live range???
    hsa_ext_control_directives_t control_directives;
    memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));

    err = hsa_ext_program_finalize(DeviceInfo.hsaProgram, isa, 0, control_directives, "", HSA_CODE_OBJECT_TYPE_PROGRAM, &code_object);
    check(Finalizing the program, err);

    /*
     * Create the empty executable.
     */
    err = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);
    check(Create the executable, err);

  }
  else {
    /*
     * Create the empty executable.
     */
    err = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &executable);
    check(Create the executable, err);

    /*
     * Deserialize.
     */
    //FIXME: HSA runtime need malloc to see the right address range
    void *new_img = malloc(img_size);
    memcpy(new_img, image->ImageStart, img_size);
    //err = hsa_code_object_deserialize((char *)image->ImageStart, img_size, NULL, &code_object);
    err = hsa_code_object_deserialize((char *)new_img, img_size, NULL, &code_object);
    check(Deserialize, err);
    free(new_img);
    // printf("deserialization return code: %d\n", err);

    // extra check
    assert(code_object.handle != 0);
  }

  /*
   * Load the code object.
   */
  err = hsa_executable_load_code_object(executable, DeviceInfo.agent, code_object, "");
  check(Loading the code object, err);

  /*
   * Freeze the executable; it can now be queried for symbols.
   */
  // FIXME: live range???
  err = hsa_executable_freeze(executable, "");
  check(Freeze the executable, err);

  char kernel_name[128] = "&__OpenCL_vector_copy_kernel";
  char name_buffer[128];

  // Here, we take advantage of the data that is appended after img_end to get
  // the symbols' name we need to load. This data consist of the host entries
  // begin and end as well as the target name (see the offloading linker script
  // creation in clang compiler).
  // Find the symbols in the module by name. The name can be obtain by
  // concatenating the host entry name with the target name

  __tgt_offload_entry *HostBegin = image->EntriesBegin;
  __tgt_offload_entry *HostEnd   = image->EntriesEnd;

  for( __tgt_offload_entry *e = HostBegin; e != HostEnd; ++e) {
    /*
     * Execution symbol
     */
    hsa_executable_symbol_t * symbol = (hsa_executable_symbol_t *)malloc(sizeof(hsa_executable_symbol_t))  ;

    if( !e->addr ){
      // FIXME: Probably we should fail when something like this happen, the
      // host should have always something in the address to uniquely identify
      // the target region.
      DP("Analyzing host entry '<null>' (size = %lld)...\n",
          (unsigned long long)e->size);

      __tgt_offload_entry entry = *e;
      DeviceInfo.addOffloadEntry(device_id, entry);
      continue;
    }

    if( e->size ){

      __tgt_offload_entry entry = *e;

      DP("Find globale var name: %s\n", e->name);


#if 0
      CUdeviceptr cuptr;
      size_t cusize;
      err = cuModuleGetGlobal(&cuptr,&cusize,cumod,e->name);

      if (err != CUDA_SUCCESS){
        DP("loading global '%s' (Failed)\n",e->name);
        CUDA_ERR_STRING (err);
        return NULL;
      }

      if ((int32_t)cusize != e->size){
        DP("loading global '%s' - size mismatch (%lld != %lld)\n",e->name,
            (unsigned long long)cusize,
            (unsigned long long)e->size);
        CUDA_ERR_STRING (err);
        return NULL;
      }

      DP("Entry point %ld maps to global %s (%016lx)\n",e-HostBegin,e->name,(long)cuptr);
      entry.addr = (void*)cuptr;

      DeviceInfo.addOffloadEntry(device_id, entry);

#endif
      continue;
    }

    /*
     * Extract the symbol from the executable.
     */
    if (useBrig) {
      sprintf(name_buffer, "&%s", (e->name));
      DP("to find the kernel name: %s\n", name_buffer);
      err = hsa_executable_get_symbol(executable, NULL, name_buffer, DeviceInfo.agent, 0, symbol);
      check(Extract the symbol from the executable, err);

      if (err != HSA_STATUS_SUCCESS) {
        sprintf(name_buffer, "&%s", (e->name+1));
        DP("try another kernel name: %s\n", name_buffer);
        err = hsa_executable_get_symbol(executable, NULL, name_buffer, DeviceInfo.agent, 0, symbol);
        check(Extract the symbol from the executable, err);
      }
    }
    else {
      sprintf(name_buffer, "%s", e->name);
      DP("to find the kernel name: %s\n", name_buffer);
      err = hsa_executable_get_symbol(executable, NULL, name_buffer, DeviceInfo.agent, 0, symbol);
      check(Extract the symbol from the executable, err);

      if (err != HSA_STATUS_SUCCESS) {
        sprintf(name_buffer, "%s", (e->name+1));
        DP("try another kernel name: %s\n", name_buffer);
        err = hsa_executable_get_symbol(executable, NULL, name_buffer, DeviceInfo.agent, 0, symbol);
        check(Extract the symbol from the executable, err);
      }
    }

    if (err != HSA_STATUS_SUCCESS) {
      DP("TO TEST: cp vector_copy.wrap .so.tgt-hsail64\n");
      err = hsa_executable_get_symbol(executable, NULL, kernel_name, DeviceInfo.agent, 0, symbol);
      check(Extract the symbol from the executable, err);
    }

    /*
     * Get the hsa symbol address.
     */

    // we malloc, need free

    /*
     * Register the hsa symbol address.
     */
    if (err == HSA_STATUS_SUCCESS) {
      __tgt_offload_entry entry = *e;
      entry.addr = (void*)symbol;
      DP("Register target offloading entry point at %016llx.\n",(long long unsigned)(Elf64_Addr)symbol);
      DeviceInfo.addOffloadEntry(device_id, entry);
    }
  }

  return DeviceInfo.getOffloadEntriesTable(device_id);
}

void *__tgt_rtl_data_alloc(int device_id, int size){
  // HSA runtime
  hsa_status_t err;

  /*
   * Alloc system memory on host
   */
  void *ptr = malloc(size);
  //FIXME: do we need unregister???
  err=hsa_memory_register(ptr, size);
  check(Registering hsa memory, err);

  DeviceInfo.addMemory(device_id,ptr,size);
  DP("System Alloc data %d bytes, (tgt:%016llx).\n", size, (long long unsigned)(Elf64_Addr)ptr);
  return ptr;
}

int32_t __tgt_rtl_data_submit(int device_id, void *tgt_ptr, void *hst_ptr, int size){
  // HSA runtime
  hsa_status_t err;

  DP("Submit data %d bytes, (hst:%016llx) -> (tgt:%016llx).\n", size, (long long unsigned)(Elf64_Addr)hst_ptr, (long long unsigned)(Elf64_Addr)tgt_ptr);

  bool valid = DeviceInfo.findMemory(device_id,tgt_ptr);
  assert( valid && "Using invalid target pointer" );

  if(!valid)
    return OFFLOAD_FAIL;

  /*
   * Extra system memory copy on host
   */
  memcpy(tgt_ptr,hst_ptr,size);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int device_id, void *hst_ptr, void *tgt_ptr, int size){
  // HSA runtime
  hsa_status_t err;

  bool valid = DeviceInfo.findMemory(device_id,tgt_ptr);
  assert( valid && "Using invalid target pointer" );

  if(!valid)
    return OFFLOAD_FAIL;

  /*
   * Extra system memory copy on host
   */
  memcpy(hst_ptr,tgt_ptr,size);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int device_id, void* tgt_ptr) {
  // HSA runtime
  hsa_status_t err;

  bool valid = DeviceInfo.findMemory(device_id,tgt_ptr);
  assert( valid && "Using invalid target pointer" );

  if(!valid)
    return OFFLOAD_FAIL;

  DeviceInfo.deleteMemory(device_id,tgt_ptr);

  //need size?
  //From HSA team we can put 0 there.
  err=hsa_memory_deregister(tgt_ptr, 0);
  check(Deregistering hsa memory, err);

  /*
   * Free system memory on host
   */
  free(tgt_ptr);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id,
    void *tgt_entry_ptr, void **tgt_args, int32_t arg_num,
    int32_t team_num, int32_t thread_limit) {

  hsa_status_t err;

  // Set the context we are using
  // update thread limit content in gpu memory if un-initialized or specified from host

  /*
   * Set limit based on ThreadsPerGroup and GroupsPerDevice
   */
  int threadsPerGroup;

  if (thread_limit>0 && thread_limit <= DeviceInfo.ThreadsPerGroup[device_id]) {
    threadsPerGroup=thread_limit;
  }
  else {
    // by default we use wavefront size as the thread limit
    threadsPerGroup=WAVEFRONTSIZE;
  }

  // The current OpenMP spec only
  // allows thread_limit clause available on teams, not on target

  int groupsPerDevice;

  if (team_num > 0 && team_num <= DeviceInfo.GroupsPerDevice[device_id]) {
    groupsPerDevice=team_num;
  }
  else {
    groupsPerDevice=DeviceInfo.GroupsPerDevice[device_id];
  }

  DP("Device %d: runtime size for groupsPerDevice %d & threadsPerGroup %d\n",
      device_id, groupsPerDevice, threadsPerGroup);

  DP("Device %d: total threads %d x %d = %d\n",
      device_id,
      threadsPerGroup,
      groupsPerDevice,
      threadsPerGroup * groupsPerDevice);

  int local_size=threadsPerGroup;
  int global_size=threadsPerGroup*groupsPerDevice;

  /*
   * Get back the registered target address.
   */
  hsa_executable_symbol_t *symbol = (hsa_executable_symbol_t *) tgt_entry_ptr ;

  DP("Address from the host %016llx.\n",(long long unsigned)(Elf64_Addr)tgt_entry_ptr);

  // Run on the device
  /*
   * Create a signal to wait for the dispatch to finish.
   */
  hsa_signal_t signal;
  err=hsa_signal_create(1, 0, NULL, &signal);
  check(Creating a HSA signal, err);

  /*
   * Initialize the dispatch packet.
   */
  /*
   * Extract dispatch information from the symbol
   */
  uint64_t kernel_object;
  uint32_t kernarg_segment_size;
  uint32_t group_segment_size;
  uint32_t private_segment_size;

  err = hsa_executable_symbol_get_info(*symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  check(Extracting the symbol from the executable, err);

  err = hsa_executable_symbol_get_info(*symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_segment_size);
  check(Extracting the kernarg segment size from the executable, err);
  DP("kernarg_segment_size: %d\n", kernarg_segment_size);

  err = hsa_executable_symbol_get_info(*symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_segment_size);
  check(Extracting the group segment size from the executable, err);
  DP("group_segment_size: %d\n", group_segment_size);

  err = hsa_executable_symbol_get_info(*symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_segment_size);
  check(Extracting the private segment from the executable, err);
  DP("private_segment_size: %d\n", private_segment_size);


  // All args are references, check before kernel
  for(int32_t i=0; i<arg_num; ++i) {
    if (i!= arg_num-1) {
      DP("Arg %d, first element as int: (tgt:%016llx) %d.\n", i, (long long unsigned)(Elf64_Addr)tgt_args[i], *(int *)tgt_args[i]);
    }
  }

#ifdef RUNTIME_TEST
  /*
   * Allocate and initialize the kernel arguments and data.
   */

  uint64_t total_buffer_size = global_size*4;

  char* arg1=(char*)malloc(global_size*4);
  //memset(arg1, 1, global_size*4);
  *(int *)arg1 = 1;
  err=hsa_memory_register(arg1, global_size*4);
  check(Registering argument memory for input parameter, err);

  char* arg2=(char*)malloc(global_size*4);
  //memset(arg2, 0, global_size*4);
  *(int *)arg2 = 0;
  err=hsa_memory_register(arg2, global_size*4);
  check(Registering argument memory for output parameter, err);


  // Notice the order
  // arg1=(char *)tgt_args[1];
  // arg2=(char *)tgt_args[0];

  struct __attribute__ ((aligned(16))) args_t {
    void* arg1;
    void* arg2;
  } hostArgs;

  memset(&hostArgs, 0, sizeof(hostArgs));

  hostArgs.arg1=arg1;
  hostArgs.arg2=arg2;
#endif

  /*
   * Find a memory region that supports kernel arguments.
   */
  hsa_region_t kernarg_region;
  kernarg_region.handle=(uint64_t)-1;
  hsa_agent_iterate_regions(DeviceInfo.agent, get_kernarg_memory_region, &kernarg_region);
  err = (kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
  check(Finding a kernarg memory region, err);

  /*
   * Allocate the kernel argument buffer from the correct region.
   */
  void* kernarg_address = NULL;
  err = hsa_memory_allocate(kernarg_region, kernarg_segment_size, &kernarg_address);
  check(Allocating kernel argument memory buffer, err);

  /*
   * Fill the kernarg_region
   */
#ifdef RUNTIME_TEST
  memcpy(kernarg_address, &hostArgs, sizeof(hostArgs));
#else
  memcpy(kernarg_address, tgt_args, sizeof(void*)*arg_num);
#endif

  /*
   * Obtain the current queue write index.
   */
  uint64_t index = hsa_queue_load_write_index_relaxed(DeviceInfo.commandQueue);

  /*
   * Write the aql packet at the calculated queue index address.
   */
  const uint32_t queueMask = DeviceInfo.commandQueue->size - 1;
  hsa_kernel_dispatch_packet_t* dispatch_packet = &(((hsa_kernel_dispatch_packet_t*)(DeviceInfo.commandQueue->base_address))[index&queueMask]);

  dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  dispatch_packet->header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  dispatch_packet->setup  |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch_packet->workgroup_size_x = (uint16_t) (local_size);
  dispatch_packet->workgroup_size_y = (uint16_t)1;
  dispatch_packet->workgroup_size_z = (uint16_t)1;
  dispatch_packet->grid_size_x = (uint32_t) (global_size);
  dispatch_packet->grid_size_y = 1;
  dispatch_packet->grid_size_z = 1;
  dispatch_packet->completion_signal = signal;
  dispatch_packet->kernel_object = kernel_object;
  dispatch_packet->kernarg_address = (void*) kernarg_address;
  dispatch_packet->private_segment_size = private_segment_size;
  dispatch_packet->group_segment_size = group_segment_size;
  __atomic_store_n((uint8_t*)(&dispatch_packet->header), (uint8_t)HSA_PACKET_TYPE_KERNEL_DISPATCH, __ATOMIC_RELEASE);

  /*
   * Increment the write index and ring the doorbell to dispatch the kernel.
   */
  hsa_queue_store_write_index_relaxed(DeviceInfo.commandQueue, index+1);
  hsa_signal_store_relaxed(DeviceInfo.commandQueue->doorbell_signal, index);
  check(Dispatching the kernel, err);

  /*
   * Wait on the dispatch signal until the kernel is finished.
   */
  hsa_signal_value_t value = hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  /*
   * Validate the data in the output buffer.
   */
#ifdef RUNTIME_VERIFY
  int valid = memcmp(tgt_args[0],tgt_args[1], sizeof(int));
  if(!valid) {
    printf("Passed runtime validation.\n");
  } else {
    printf("Failed runtime Validation %d!\n", valid);
  }
#endif

#ifdef RUNTIME_TEST
  free(arg1);
  free(arg2);
#endif

  // All args are references, check after kernel
  for(int32_t i=0; i<arg_num; ++i) {
    if (i!= arg_num-1) {
      DP("Arg %d, first element as int: (tgt:%016llx) %d.\n", i, (long long unsigned)(Elf64_Addr)tgt_args[i], *(int *)tgt_args[i]);
    }
  }

  /*
   * Cleanup all allocated resources.
   */
  err = hsa_memory_free(kernarg_address);
  check(Freeing kernel argument memory buffer, err);

  err=hsa_signal_destroy(signal);
  check(Destroying the signal, err);

  return OFFLOAD_SUCCESS;

}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
    void **tgt_args, int32_t arg_num)
{
  // use one team and one thread
  // fix thread num
  int32_t team_num = 1;
  int32_t thread_limit = 0; // use default
  return __tgt_rtl_run_target_team_region(device_id,
      tgt_entry_ptr, tgt_args, arg_num, team_num, thread_limit);
}

#ifdef __cplusplus
}
#endif

