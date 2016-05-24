//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <dlfcn.h>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <queue>
#include <mutex>

// Header file global to this project
#include "omptarget.h"

#define DP(...) DEBUGP("Libomptarget", __VA_ARGS__)

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* PowerPC target */ "libomptarget.rtl.ppc64.so",
    /* x86_64 target  */ "libomptarget.rtl.x86_64.so",
    /* AMD HSA target  */ "libomptarget.rtl.hsa.so",
    /* CUDA target    */ "libomptarget.rtl.cuda.so"};

struct RTLInfoTy;

/// Map between host data and target data.
struct HostDataToTargetTy {
  long HstPtrBase; // host info.
  long HstPtrBegin;
  long HstPtrEnd; // non-inclusive.

  long TgtPtrBegin; // target info.
  long TgtPtrEnd;   // non-inclusive (FIXME: maybe not needed?)

  long RefCount;

  HostDataToTargetTy()
      : HstPtrBase(0), HstPtrBegin(0), HstPtrEnd(0), TgtPtrBegin(0),
        TgtPtrEnd(0), RefCount(0) {}
  HostDataToTargetTy(long BP, long B, long E, long TB, long TE)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E), TgtPtrBegin(TB),
        TgtPtrEnd(TE), RefCount(1) {}
};

typedef std::list<HostDataToTargetTy> HostDataToTargetListTy;

struct DeviceTy {
  int32_t DeviceID;
  RTLInfoTy *RTL;
  int32_t RTLDeviceID;

  bool IsInit;
  HostDataToTargetListTy HostDataToTargetMap;
  std::list<void *> PendingConstrDestrHostPtrList;

  DeviceTy(RTLInfoTy *RTL)
      : DeviceID(-1), RTL(RTL), RTLDeviceID(-1), IsInit(false),
        HostDataToTargetMap(), PendingConstrDestrHostPtrList() {}

  void *getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase, long Size,
                         long &IsNew, long UpdateRefCount = true);
  void *getTgtPtrBegin(void *HstPtrBegin, long Size);
  void *getTgtPtrBegin(void *HstPtrBegin, long Size, long &IsLast,
                       long UpdateRefCount = true);
  void deallocTgtPtr(void *TgtPtrBegin, long Size, long ForceDelete);

  // calls to RTL
  int32_t init();
  __tgt_target_table *load_binary(void *Img);

  int32_t data_submit(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size);
  int32_t data_retrieve(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);

  int32_t run_region(void *TgtEntryPtr, void **TgtVarsPtr, int32_t TgtVarsSize);
  int32_t run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
                          int32_t TgtVarsSize, int32_t NumTeams,
                          int32_t ThreadLimit);
};

struct RTLInfoTy {
  typedef int32_t(is_valid_binary_ty)(void *);
  typedef int32_t(number_of_devices_ty)();
  typedef int32_t(init_device_ty)(int32_t);
  typedef __tgt_target_table *(load_binary_ty)(int32_t, void *);
  typedef void *(data_alloc_ty)(int32_t, int64_t);
  typedef int32_t(data_submit_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_retrieve_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_delete_ty)(int32_t, void *);
  typedef int32_t(run_region_ty)(int32_t, void *, void **, int32_t);
  typedef int32_t(run_team_region_ty)(int32_t, void *, void **, int32_t,
                                      int32_t, int32_t);

  int32_t Idx;                     // RTL index, index is the number of devices
                                   // of other RTLs that were registered before.
  int32_t NumberOfDevices;         // Number of devices this RTL deal with.
  std::vector<DeviceTy *> Devices; // one per device (NumberOfDevices in total).

  void *LibraryHandler;
  // Functions implemented in the RTL.
  is_valid_binary_ty *is_valid_binary;
  number_of_devices_ty *number_of_devices;
  init_device_ty *init_device;
  load_binary_ty *load_binary;
  data_alloc_ty *data_alloc;
  data_submit_ty *data_submit;
  data_retrieve_ty *data_retrieve;
  data_delete_ty *data_delete;
  run_region_ty *run_region;
  run_team_region_ty *run_team_region;

  // Are there images associated with this RTL.
  bool isUsed;
};

/// Map between Device ID (i.e. openmp device id) and its DeviceTy.
typedef std::vector<DeviceTy> DevicesTy;
static DevicesTy Devices;

/// RTLs identified in the system.
class RTLsTy {
  // Set to true if the library attempted to load the RTLs (plugins) before.
  bool isInitialized;

public:
  // List of the detected runtime libraries.
  std::list<RTLInfoTy> AllRTLs;

  // Array of pointers to the detected runtime libraries that have compatible
  // binaries.
  std::vector<RTLInfoTy *> UsedRTLs;

  explicit RTLsTy() : isInitialized(false) {}

  // Load all the runtime libraries (plugins) if not done before.
  void LoadRTLs();
};

void RTLsTy::LoadRTLs() {
  // Did we load the RTLs before? If so, just return.
  if (isInitialized)
    return;

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (auto *Name : RTLNames) {
    void *dynlib_handle = dlopen(Name, RTLD_NOW);

    if (!dynlib_handle) {
      // Library does not exist or cannot be found.
      DP("Unable to load library '%s': %s!\n", Name, dlerror());
      continue;
    }

    DP("Successfully loaded library '%s'!\n", Name);

    // Retrieve the RTL information from the runtime library.
    RTLInfoTy R;

    R.LibraryHandler = dynlib_handle;
    R.isUsed = false;
    if (!(R.is_valid_binary = (RTLInfoTy::is_valid_binary_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_is_valid_binary")))
      continue;
    if (!(R.number_of_devices = (RTLInfoTy::number_of_devices_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_number_of_devices")))
      continue;
    if (!(R.init_device = (RTLInfoTy::init_device_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_init_device")))
      continue;
    if (!(R.load_binary = (RTLInfoTy::load_binary_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_load_binary")))
      continue;
    if (!(R.data_alloc = (RTLInfoTy::data_alloc_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_alloc")))
      continue;
    if (!(R.data_submit = (RTLInfoTy::data_submit_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_submit")))
      continue;
    if (!(R.data_retrieve = (RTLInfoTy::data_retrieve_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_retrieve")))
      continue;
    if (!(R.data_delete = (RTLInfoTy::data_delete_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_data_delete")))
      continue;
    if (!(R.run_region = (RTLInfoTy::run_region_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_run_target_region")))
      continue;
    if (!(R.run_team_region = (RTLInfoTy::run_team_region_ty *)dlsym(
              dynlib_handle, "__tgt_rtl_run_target_team_region")))
      continue;

    // No devices are supported by this RTL?
    if (!(R.NumberOfDevices = R.number_of_devices())) {
      DP("No devices supported in this RTL\n");
      continue;
    }

    DP("Registering RTL %016lx supporting %d devices!\n", (long)dynlib_handle,
       R.NumberOfDevices);

    // The RTL is valid! Will save the information in the RTLs list.
    AllRTLs.push_back(R);
  }
  return;
}

static RTLsTy RTLs;

/// Map between the host entry begin and the translation table. Each
/// registered library gets one TranslationTable. Use the map from
/// __tgt_offload_entry so that we may quickly determine if we are
/// trying to (re)register an existing lib, or if we really have a new one.
struct TranslationTable {
  __tgt_target_table HostTable;

  // Image assigned to a given device.
  std::vector<__tgt_device_image *> TargetsImages; // One image per device ID.

  // Table of entry points or NULL if it was not already computed.
  std::vector<__tgt_target_table *> TargetsTable; // One table per device ID.
};
typedef std::map<__tgt_offload_entry *, TranslationTable>
    HostEntriesBeginToTransTableTy;
static HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;

/// Map between the host ptr and a table index
struct TableMap {
  TranslationTable *Table; // table associated with the host ptr.
  uint32_t Index; // index in which the host ptr translated entry is found.
  TableMap() : Table(0), Index(0) {}
  TableMap(TranslationTable *table, uint32_t index)
      : Table(table), Index(index) {}
};
typedef std::map<void *, TableMap> HostPtrToTableMapTy;
static HostPtrToTableMapTy HostPtrToTableMap;

////////////////////////////////////////////////////////////////////////////////
// getter and setter
//
// FIXME: Non-compliant. This need to be integrated in KMPC; can keep it
// here for the moment

static int DefaultDevice = 0;

void omp_set_default_device(int device_num) { DefaultDevice = device_num; }

int omp_get_default_device(void) { return DefaultDevice; }

int omp_get_num_devices(void) { return Devices.size(); }

int omp_is_initial_device(void) { return true; }

////////////////////////////////////////////////////////////////////////////////
// functionality for device

// return the target pointer begin (where the data will be moved).
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, long Size) {
  long IsLast;
  return getTgtPtrBegin(HstPtrBegin, Size, IsLast, false);
}
void *DeviceTy::getTgtPtrBegin(void *HstPtrBegin, long Size, long &IsLast,
                               long UpdateRefCount) {
  long hp = (long)HstPtrBegin;
  IsLast = false;

  for (auto &HT : HostDataToTargetMap) {
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      if ((hp + Size) > HT.HstPtrEnd) {
        DP("WARNING: Array contain pointer but does not contain the complete "
           "section\n");
      }

      IsLast = !(HT.RefCount > 1);

      if (HT.RefCount > 1 && UpdateRefCount)
        --HT.RefCount;

      long tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
      return (void *)tp;
    }
  }

  return NULL;
}

// return the target pointer begin (where the data will be moved).
void *DeviceTy::getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase, long Size,
                                 long &IsNew, long UpdateRefCount) {
  long hp = (long)HstPtrBegin;
  IsNew = false;

  // Check if the pointer is contained.
  for (auto &HT : HostDataToTargetMap) {
    // Is it contained?
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      if ((hp + Size) > HT.HstPtrEnd) {
        DP("WARNING: Array contain pointer but does not contain the complete "
           "section\n");
      }
      if (UpdateRefCount)
        ++HT.RefCount;
      long tp = HT.TgtPtrBegin + (hp - HT.HstPtrBegin);
      return (void *)tp;
    }
  }

  // It is not contained we should create a new entry for it.
  IsNew = true;
  long tp = (long)RTL->data_alloc(RTLDeviceID, Size);
  HostDataToTargetMap.push_front(
      HostDataToTargetTy((long)HstPtrBase, hp, hp + Size, tp, tp + Size));
  return (void *)tp;
}

void DeviceTy::deallocTgtPtr(void *HstPtrBegin, long Size, long ForceDelete) {
  long hp = (long)HstPtrBegin;

  // Check if the pointer is contained in any sub-nodes.
  for (auto ii = HostDataToTargetMap.begin(), ie = HostDataToTargetMap.end();
       ii != ie; ++ii) {
    auto &HT = *ii;
    // Is it contained?
    if (hp >= HT.HstPtrBegin && hp < HT.HstPtrEnd) {
      if ((hp + Size) > HT.HstPtrEnd) {
        DP("WARNING: Array contain pointer but does not contain the complete "
           "section\n");
      }
      if (ForceDelete)
        HT.RefCount = 1;
      if (--HT.RefCount <= 0) {
        assert(HT.RefCount == 0 && "did not expect a negative ref count");
        DP("Deleting tgt data 0x%016llx of size %lld\n",
           (long long)HT.TgtPtrBegin, (long long)Size);
        RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
        HostDataToTargetMap.erase(ii);
      }
      return;
    }
  }
  DP("Section to delete (hst addr 0x%llx) does not exist in the allocated "
     "memory\n",
     (unsigned long long)hp);
}

// init device.
int32_t DeviceTy::init() {
  int32_t rc = RTL->init_device(RTLDeviceID);
  if (rc == OFFLOAD_SUCCESS) {
    IsInit = true;
  }
  return rc;
}

// load binary to device.
__tgt_target_table *DeviceTy::load_binary(void *Img) {
  return RTL->load_binary(RTLDeviceID, Img);
}

// submit data to device.
int32_t DeviceTy::data_submit(void *TgtPtrBegin, void *HstPtrBegin,
                              int64_t Size) {
  return RTL->data_submit(RTLDeviceID, TgtPtrBegin, HstPtrBegin, Size);
}

// retrieve data from device.
int32_t DeviceTy::data_retrieve(void *HstPtrBegin, void *TgtPtrBegin,
                                int64_t Size) {
  return RTL->data_retrieve(RTLDeviceID, HstPtrBegin, TgtPtrBegin, Size);
}

// run region on device
int32_t DeviceTy::run_region(void *TgtEntryPtr, void **TgtVarsPtr,
                             int32_t TgtVarsSize) {
  return RTL->run_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtVarsSize);
}

// run team region on device.
int32_t DeviceTy::run_team_region(void *TgtEntryPtr, void **TgtVarsPtr,
                                  int32_t TgtVarsSize, int32_t NumTeams,
                                  int32_t ThreadLimit) {
  return RTL->run_team_region(RTLDeviceID, TgtEntryPtr, TgtVarsPtr, TgtVarsSize,
                              NumTeams, ThreadLimit);
}

////////////////////////////////////////////////////////////////////////////////
// functionality for registering libs

static void RegisterImageIntoTranslationTable(TranslationTable &TT,
                                              RTLInfoTy &RTL,
                                              __tgt_device_image *image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = RTL.Idx + RTL.NumberOfDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t i = 0; i < RTL.NumberOfDevices; ++i) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.Idx + i] != image) {
      TT.TargetsImages[RTL.Idx + i] = image;
      TT.TargetsTable[RTL.Idx + i] = 0; // lazy initialization of target table.
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {

  // Attempt to load all the plugins available in the system.
  RTLs.LoadRTLs();

  // Register the images with the RTLs that understand them, if any.
  for (int32_t i = 0; i < desc->NumDevices; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    bool FoundRTL = false;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We attempt to use the RTLs that are already being
    // used first.
    for (auto *R : RTLs.UsedRTLs) {

      assert(R->isUsed && "Expecting used RTLs.");

      if (!R->is_valid_binary(img)) {
        DP("Image %016lx is NOT compatible with RTL %016lx!\n",
           (long)img->ImageStart, (long)R->LibraryHandler);
        continue;
      }

      DP("Image %016lx is compatible with RTL %016lx!\n", (long)img->ImageStart,
         (long)R->LibraryHandler);

      // Initialize translation table for this.
      TranslationTable &TransTable =
          HostEntriesBeginToTransTable[desc->EntriesBegin];
      TransTable.HostTable.EntriesBegin = desc->EntriesBegin;
      TransTable.HostTable.EntriesEnd = desc->EntriesEnd;

      DP("Registering image %016lx with RTL %016lx!\n", (long)img->ImageStart,
         (long)R->LibraryHandler);
      RegisterImageIntoTranslationTable(TransTable, *R, img);
      FoundRTL = true;
      break;
    }

    if (FoundRTL)
      break;

    DP("No RTLs in use support the image %016lx!\n", (long)img->ImageStart);

    // Find a compatible RTL that is not being used.
    for (auto &R : RTLs.AllRTLs) {
      if (R.isUsed)
        continue;

      if (!R.is_valid_binary(img)) {
        DP("Image %016lx is NOT compatible with RTL %016lx!\n",
           (long)img->ImageStart, (long)R.LibraryHandler);
        continue;
      }

      DP("Image %016lx is compatible with RTL %016lx!\n", (long)img->ImageStart,
         (long)R.LibraryHandler);

      // Initialize the device information for the RTL we are about to use.
      DeviceTy device(&R);

      size_t start = Devices.size();
      Devices.resize(start + R.NumberOfDevices, device);
      for (int32_t device_id = 0; device_id < R.NumberOfDevices; device_id++) {
        // global device ID
        Devices[start + device_id].DeviceID = start + device_id;
        // RTL local device ID
        Devices[start + device_id].RTLDeviceID = device_id;

        // Save pointer to device in RTL in case we want to unregister the RTL
        R.Devices.push_back(&Devices[start + device_id]);
      }

      // Initialize the index of this RTL and save it in the used RTLs.
      R.Idx = (RTLs.UsedRTLs.empty())
                  ? 0
                  : RTLs.UsedRTLs.back()->Idx +
                        RTLs.UsedRTLs.back()->NumberOfDevices;
      R.isUsed = true;
      RTLs.UsedRTLs.push_back(&R);

      DP("RTL %016lx has index %d!\n", (long)R.LibraryHandler, R.Idx);

      // Initialize translation table for this.
      TranslationTable &TransTable =
          HostEntriesBeginToTransTable[desc->EntriesBegin];
      TransTable.HostTable.EntriesBegin = desc->EntriesBegin;
      TransTable.HostTable.EntriesEnd = desc->EntriesEnd;

      DP("Registering image %016lx with RTL %016lx!\n", (long)img->ImageStart,
         (long)R.LibraryHandler);
      RegisterImageIntoTranslationTable(TransTable, R, img);
      FoundRTL = true;
      break;
    }

    if (FoundRTL)
      break;

    DP("No RTL found for image %016lx!\n", (long)img->ImageStart);
  }

  DP("Done register entries!\n");
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  DP("Unloading target library!\n");
  return;
}

/// Internal function to do the mapping and transfer the data to the device
static void target_data_begin(DeviceTy &Device, int32_t arg_num,
                              void **args_base, void **args, int64_t *arg_sizes,
                              int32_t *arg_types) {
  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    void *Pointer_TgtPtrBegin;
    long IsNew, Pointer_IsNew;
    if (arg_types[i] & tgt_map_pointer) {
      DP("has a pointer entry: \n");
      // base is address of pointer.
      Pointer_TgtPtrBegin = Device.getOrAllocTgtPtr(
          HstPtrBase, HstPtrBase, sizeof(void *), Pointer_IsNew);
      DP("There are %ld bytes allocated at target address %016lx\n",
         (long)sizeof(void *), (long)Pointer_TgtPtrBegin);
      assert(Pointer_TgtPtrBegin &&
             "Data allocation by RTL returned invalid ptr");
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
    }

    void *TgtPtrBegin =
        Device.getOrAllocTgtPtr(HstPtrBegin, HstPtrBase, arg_sizes[i], IsNew);
    DP("There are %ld bytes allocated at target address %016lx - is new %ld\n",
       (long)arg_sizes[i], (long)TgtPtrBegin, IsNew);
    assert(TgtPtrBegin && "Data allocation by RTL returned invalid ptr");

    if ((arg_types[i] & tgt_map_to) &&
        (IsNew || (arg_types[i] & tgt_map_always))) {
      DP("Moving %ld bytes (hst:%016lx) -> (tgt:%016lx)\n", (long)arg_sizes[i],
         (long)HstPtrBegin, (long)TgtPtrBegin);
      Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
    }

    if (arg_types[i] & tgt_map_pointer) {
      DP("Update pointer (%016lx) -> [%016lx]\n", (long)Pointer_TgtPtrBegin,
         (long)TgtPtrBegin);
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *TgrPtrBase_Value = (void *)((uint64_t)TgtPtrBegin - Delta);
      Device.data_submit(Pointer_TgtPtrBegin, &TgrPtrBase_Value,
                         sizeof(void *));
    }
  }
}

EXTERN void __tgt_target_data_begin_nowait(int32_t device_id, int32_t arg_num,
                                           void **args_base, void **args,
                                           int64_t *arg_sizes,
                                           int32_t *arg_types, int32_t depNum,
                                           void *depList, int32_t noAliasDepNum,
                                           void *noAliasDepList) {
  __tgt_target_data_begin(device_id, arg_num, args_base, args, arg_sizes,
                          arg_types);
}

/// creates host-to-target data mapping, store it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int32_t device_id, int32_t arg_num,
                                    void **args_base, void **args,
                                    int64_t *arg_sizes, int32_t *arg_types) {
  DP("Entering data begin region for device %d with %d mappings\n", device_id,
     arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
    DP("Use default device id %d\n", device_id);
  }
  if (Devices.size() <= (size_t)device_id) {
    DP("Device ID  %d does not have a matching RTL.\n", device_id);
    return;
  }

  // Get device info
  DeviceTy &Device = Devices[device_id];
  // Init the device if not done before
  if (!Device.IsInit) {
    if (Device.init() != OFFLOAD_SUCCESS) {
      DP("failed to init device %d\n", device_id);
      return;
    }
  }

  target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);
}

/// Internal function to undo the mapping and retrieve the data from the device.
static void target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
                            void **args, int64_t *arg_sizes,
                            int32_t *arg_types) {
  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    long IsLast;
    long ForceDelete = arg_types[i] & tgt_map_delete;
    if (arg_types[i] & tgt_map_pointer) {
      // base is pointer begin.
      Device.getTgtPtrBegin(HstPtrBase, sizeof(void *), IsLast);
      if (IsLast || ForceDelete) {
        Device.deallocTgtPtr(HstPtrBase, sizeof(void *), ForceDelete);
      }
    }
    void *TgtPtrBegin =
        Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast);

    DP("There are %ld bytes allocated at target address %016lx - is last %ld\n",
       (long)arg_sizes[i], (long)TgtPtrBegin, IsLast);

    long Always = arg_types[i] & tgt_map_always;
    if ((arg_types[i] & tgt_map_from) && (IsLast || ForceDelete || Always)) {
      DP("Moving %ld bytes (tgt:%016lx) -> (hst:%016lx)\n", (long)arg_sizes[i],
         (long)TgtPtrBegin, (long)HstPtrBegin);
      Device.data_retrieve(HstPtrBegin, TgtPtrBegin, arg_sizes[i]);
    }
    if (IsLast || ForceDelete) {
      Device.deallocTgtPtr(HstPtrBegin, arg_sizes[i], ForceDelete);
    }
  }
}

/// passes data from the target, release target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int32_t device_id, int32_t arg_num,
                                  void **args_base, void **args,
                                  int64_t *arg_sizes, int32_t *arg_types) {
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }
  if (Devices.size() <= (size_t)device_id) {
    DP("Device ID  %d does not have a matching RTL.\n", device_id);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("uninit device: ignore");
    return;
  }

  target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
}

EXTERN void __tgt_target_data_end_nowait(int32_t device_id, int32_t arg_num,
                                         void **args_base, void **args,
                                         int64_t *arg_sizes, int32_t *arg_types,
                                         int32_t depNum, void *depList,
                                         int32_t noAliasDepNum,
                                         void *noAliasDepList) {
  __tgt_target_data_end(device_id, arg_num, args_base, args, arg_sizes,
                        arg_types);
}

/// passes data to/from the target.
EXTERN void __tgt_target_data_update(int32_t device_id, int32_t arg_num,
                                     void **args_base, void **args,
                                     int64_t *arg_sizes, int32_t *arg_types) {
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }
  if (Devices.size() <= (size_t)device_id) {
    DP("Device ID  %d does not have a matching RTL.\n", device_id);
    return;
  }

  // Get device info.
  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("uninit device: ignore");
    return;
  }

  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    void *HstPtrBegin = args[i];
    long IsLast;
    void *TgtPtrBegin =
        Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast, false);
    if (arg_types[i] & tgt_map_from) {
      DP("Moving %ld bytes (tgt:%016lx) -> (hst:%016lx)\n", (long)arg_sizes[i],
         (long)TgtPtrBegin, (long)HstPtrBegin);
      Device.data_retrieve(HstPtrBegin, TgtPtrBegin, arg_sizes[i]);
    }
    if (arg_types[i] & tgt_map_to) {
      DP("Moving %ld bytes (hst:%016lx) -> (tgt:%016lx)\n", (long)arg_sizes[i],
         (long)HstPtrBegin, (long)TgtPtrBegin);
      Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
    }
  }
}

EXTERN void __tgt_target_data_update_nowait(
    int32_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int32_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  __tgt_target_data_update(device_id, arg_num, args_base, args, arg_sizes,
                           arg_types);
}

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// return 0 if it was able to transfer the execution to a target and an integer
/// different from zero otherwise.
static int target(int32_t device_id, void *host_ptr, int32_t arg_num,
                  void **args_base, void **args, int64_t *arg_sizes,
                  int32_t *arg_types, int32_t team_num, int32_t thread_limit,
                  int IsTeamConstruct, int IsConstrDestrRecursiveCall) {
  DP("Entering target region with entry point %016lx and device Id %d\n",
     (long)host_ptr, device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }
  // got a new constructor/destructor?
  if (device_id == OFFLOAD_DEVICE_CONSTRUCTOR ||
      device_id == OFFLOAD_DEVICE_DESTRUCTOR) {
    DP("Got a constructor/destructor\n");
    for (unsigned D = 0; D < Devices.size(); D++) {
      DeviceTy &Device = Devices[D];
      DP("device %d: enqueue constr/destr\n", D);
      Device.PendingConstrDestrHostPtrList.push_back(host_ptr);
    }
    DP("Done with constructor/destructor\n");
    return OFFLOAD_SUCCESS;
  }

  // No devices available?
  if (!(device_id >= 0 && (size_t)device_id < Devices.size())) {
    DP("Device ID %d does not have a matching RTL.\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Get device info.
  DeviceTy &Device = Devices[device_id];
  DP("Is the device %d (local is %d) initialized? %d\n", device_id,
     Device.RTLDeviceID, (int)Device.IsInit);

  // Init the device if not done before.
  if (!Device.IsInit) {
    assert(!IsConstrDestrRecursiveCall && "constr & destr should not init RT");
    if (Device.init() != OFFLOAD_SUCCESS) {
      DP("failed to init device %d\n", device_id);
      return OFFLOAD_FAIL;
    }
  }

  if (!IsConstrDestrRecursiveCall &&
      !Device.PendingConstrDestrHostPtrList.empty()) {
    DP("has pending constr/destr... call now\n");
    for (std::list<void *>::iterator
             ii = Device.PendingConstrDestrHostPtrList.begin(),
             ie = Device.PendingConstrDestrHostPtrList.end();
         ii != ie; ++ii) {
      void *ConstrDestrHostPtr = *ii;
      int rc = target(device_id, ConstrDestrHostPtr, 0, NULL, NULL, NULL, NULL,
                      1, 1, true /*team*/, true /*recursive*/);
      if (rc != OFFLOAD_SUCCESS) {
        DP("failed to run constr/destr... enqueue it\n");
        return OFFLOAD_FAIL;
      }
    }
    DP("done with pending constr/destr\n");
    Device.PendingConstrDestrHostPtrList.clear();
  }

  // Find the table information in the map or look it up in the translation
  // tables.
  TableMap *TM = 0;
  HostPtrToTableMapTy::iterator TableMapIt = HostPtrToTableMap.find(host_ptr);
  if (TableMapIt == HostPtrToTableMap.end()) {
    // We don't have a map. So search all the registered libraries.
    for (HostEntriesBeginToTransTableTy::iterator
             ii = HostEntriesBeginToTransTable.begin(),
             ie = HostEntriesBeginToTransTable.end();
         !TM && ii != ie; ++ii) {
      // get the translation table (which contains all the good info).
      TranslationTable *TransTable = &ii->second;
      // iterate over all the host table entries to see if we can locate the
      // host_ptr.
      __tgt_offload_entry *begin = TransTable->HostTable.EntriesBegin;
      __tgt_offload_entry *end = TransTable->HostTable.EntriesEnd;
      __tgt_offload_entry *cur = begin;
      for (uint32_t i = 0; cur < end; ++cur, ++i) {
        if (cur->addr != host_ptr)
          continue;
        // we got a match, now fill the HostPtrToTableMap so that we
        // may avoid this search next time.
        TM = &HostPtrToTableMap[host_ptr];
        TM->Table = TransTable;
        TM->Index = i;
        break;
      }
    }
  } else {
    TM = &TableMapIt->second;
  }
  // No map for this host pointer found!
  if (!TM) {
    DP("Host ptr %016lx does not have a matching target pointer.\n",
       (long)host_ptr);
    return OFFLOAD_FAIL;
  }

  // get target table.
  assert(TM->Table->TargetsTable.size() > (size_t)device_id &&
         "Not expecting a device ID outside the tables bounds!");
  __tgt_target_table *TargetTable = TM->Table->TargetsTable[device_id];
  // if first call, need to move the data.
  if (!TargetTable) {
    // 1) get image.
    assert(TM->Table->TargetsImages.size() > (size_t)device_id &&
           "Not expecting a device ID outside the tables bounds!");
    __tgt_device_image *img = TM->Table->TargetsImages[device_id];
    if (!img) {
      DP("No image loaded for device id %d.\n", device_id);
      return OFFLOAD_FAIL;
    }
    // 2) load image into the target table.
    TargetTable = TM->Table->TargetsTable[device_id] = Device.load_binary(img);
    // Unable to get table for this image: invalidate image and fail.
    if (!TargetTable) {
      DP("Unable to generate entries table for device id %d.\n", device_id);
      TM->Table->TargetsImages[device_id] = 0;
      return OFFLOAD_FAIL;
    }

    // Verify if the two tables sizes match.
    size_t hsize =
        TM->Table->HostTable.EntriesEnd - TM->Table->HostTable.EntriesBegin;
    size_t tsize = TargetTable->EntriesEnd - TargetTable->EntriesBegin;

    // Invalid image for this host entries!
    if (hsize != tsize) {
      DP("Host and Target tables mismatch for device id %d [%lx != %lx].\n",
         device_id, hsize, tsize);
      TM->Table->TargetsImages[device_id] = 0;
      TM->Table->TargetsTable[device_id] = 0;
      return OFFLOAD_FAIL;
    }
    assert(TM->Index < hsize &&
           "Not expecting index greater than the table size");

    // process global data that needs to be mapped.
    __tgt_target_table *HostTable = &TM->Table->HostTable;
    for (__tgt_offload_entry *CurrDeviceEntry = TargetTable->EntriesBegin,
                             *CurrHostEntry = HostTable->EntriesBegin,
                             *EntryDeviceEnd = TargetTable->EntriesEnd;
         CurrDeviceEntry != EntryDeviceEnd;
         CurrDeviceEntry++, CurrHostEntry++) {
      if (CurrDeviceEntry->size != 0) {
        // has data.
        assert(CurrDeviceEntry->size == CurrHostEntry->size &&
               "data size mismatch");
        assert(Device.getTgtPtrBegin(CurrHostEntry->addr,
                                     CurrHostEntry->size) == NULL &&
               "data in declared target should not be already mapped");
        // add entry to map.
        DP("add mapping from host 0x%llx to 0x%llx with size %lld\n\n",
           (unsigned long long)CurrHostEntry->addr,
           (unsigned long long)CurrDeviceEntry->addr,
           (unsigned long long)CurrDeviceEntry->size);
        Device.HostDataToTargetMap.push_front(HostDataToTargetTy(
            (long)CurrHostEntry->addr, (long)CurrHostEntry->addr,
            (long)CurrHostEntry->addr + CurrHostEntry->size,
            (long)CurrDeviceEntry->addr,
            (long)CurrDeviceEntry->addr + CurrDeviceEntry->size));
      }
    }
  }

  // Move data to device.
  target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);

  std::vector<void *> tgt_args;

  for (int32_t i = 0; i < arg_num; ++i) {

    if (arg_types[i] & tgt_map_extra)
      continue;

    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    void *TgtPtrBase;
    long IsLast; // unused.
    if (arg_types[i] & tgt_map_pointer) {
      DP("Obtaining target argument from host pointer %016lx to object %016lx "
         "\n",
         (long)HstPtrBase, (long)HstPtrBegin);
      void *TgtPtrBegin =
          Device.getTgtPtrBegin(HstPtrBase, sizeof(void *), IsLast, false);
      TgtPtrBase = TgtPtrBegin; // no offset for ptrs.
    } else {
      DP("Obtaining target argument from host pointer %016lx\n",
         (long)HstPtrBegin);
      void *TgtPtrBegin =
          Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast, false);
      assert(TgtPtrBegin && "NULL argument for hst ptr");
      uint64_t PtrDelta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - PtrDelta);
    }
    tgt_args.push_back(TgtPtrBase);
  }
  // Push omp handle.
  tgt_args.push_back((void *)0);

  // Launch device execution.
  int rc;
  DP("Launching target execution with pointer %016lx (index=%d).\n",
     (long)TargetTable->EntriesBegin[TM->Index].addr, TM->Index);
  if (IsTeamConstruct) {
    rc = Device.run_team_region(TargetTable->EntriesBegin[TM->Index].addr,
                                &tgt_args[0], tgt_args.size(), team_num,
                                thread_limit);
  } else {
    rc = Device.run_region(TargetTable->EntriesBegin[TM->Index].addr,
                           &tgt_args[0], tgt_args.size());
  }

  if (rc)
    return OFFLOAD_FAIL;

  // Move data from device.
  target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
  return OFFLOAD_SUCCESS;
}

EXTERN int __tgt_target(int32_t device_id, void *host_ptr, int32_t arg_num,
                        void **args_base, void **args, int64_t *arg_sizes,
                        int32_t *arg_types) {
  return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                arg_types, 0, 0, false /*team*/, false /*recursive*/);
}

EXTERN int __tgt_target_nowait(int32_t device_id, void *host_ptr,
                               int32_t arg_num, void **args_base, void **args,
                               int64_t *arg_sizes, int32_t *arg_types,
                               int32_t depNum, void *depList,
                               int32_t noAliasDepNum, void *noAliasDepList) {
  return __tgt_target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                      arg_types);
}

EXTERN int __tgt_target_teams(int32_t device_id, void *host_ptr,
                              int32_t arg_num, void **args_base, void **args,
                              int64_t *arg_sizes, int32_t *arg_types,
                              int32_t team_num, int32_t thread_limit) {
  return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                arg_types, team_num, thread_limit, true /*team*/,
                false /*recursive*/);
}

EXTERN int __tgt_target_teams_nowait(int32_t device_id, void *host_ptr,
                                     int32_t arg_num, void **args_base,
                                     void **args, int64_t *arg_sizes,
                                     int32_t *arg_types, int32_t team_num,
                                     int32_t thread_limit, int32_t depNum,
                                     void *depList, int32_t noAliasDepNum,
                                     void *noAliasDepList) {
  return __tgt_target_teams(device_id, host_ptr, arg_num, args_base, args,
                            arg_sizes, arg_types, team_num, thread_limit);
}

////////////////////////////////////////////////////////////////////////////////
// temporary for debugging (matching the ones in omptarget-nvptx

EXTERN void __kmpc_kernel_print(char *title) { DP(" %s\n", title); }

EXTERN void __kmpc_kernel_print_int8(char *title, int64_t data) {
  DP(" %s val=%lld\n", title, (long long)data);
}
