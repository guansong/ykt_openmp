//===------------- task.h - hsail OpenMP tasks support ----------- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//
//
// Task implementation support.
//
//  explicit task structure uses
//  omptarget_hsail task
//  kmp_task
//
//  where kmp_task is
//    - klegacy_TaskDescr    <- task pointer
//        shared -> X
//        routine
//        part_id
//        descr
//    -  private (of size given by task_alloc call). Accessed by
//       task+sizeof(klegacy_TaskDescr)
//        * private data *
//    - shared: X. Accessed by shared ptr in klegacy_TaskDescr
//        * pointer table to shared variables *
//    - end
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"

EXTERN kmp_TaskDescr *__kmpc_omp_task_alloc(
        kmp_Indent *loc,        // unused
        uint32_t global_tid,    // unused
        int32_t flag,           // unused (because in our impl, all are immediately exec
        size_t sizeOfTaskInclPrivate,
        size_t sizeOfSharedTable,
        kmp_TaskFctPtr taskSub)
{
    PRINT(LD_IO,
        "call __kmpc_omp_task_alloc(size priv&struct %lld, shared %lld, fct 0x%llx)\n",
        P64(sizeOfTaskInclPrivate),  P64(sizeOfSharedTable), P64(taskSub));

#if 0
    // want task+priv to be a multiple of 8 bytes
    size_t padForTaskInclPriv = PadBytes(sizeOfTaskInclPrivate, sizeof(void*));

    sizeOfTaskInclPrivate += padForTaskInclPriv;

    size_t kmpSize = sizeOfTaskInclPrivate + sizeOfSharedTable;

    ASSERT(LT_FUSSY, sizeof(omptarget_hsail_TaskDescr) % sizeof(void *) == 0,
        "need task descr of size %d to be a multiple of %d\n",
        sizeof(omptarget_hsail_TaskDescr), sizeof(void *));

    size_t totSize = sizeof(omptarget_hsail_TaskDescr) + kmpSize;

    omptarget_hsail_ExplicitTaskDescr *newExplicitTaskDescr = (omptarget_hsail_ExplicitTaskDescr *)

    SafeMalloc(totSize, "explicit task descriptor");

    kmp_TaskDescr *newKmpTaskDescr = & newExplicitTaskDescr->kmpTaskDescr;

    ASSERT0(LT_FUSSY, (uint64_t) newKmpTaskDescr ==
            (uint64_t) ADD_BYTES(newExplicitTaskDescr, sizeof(omptarget_hsail_TaskDescr)),
            "bad size assumptions");

    // init kmp_TaskDescr
    newKmpTaskDescr->sharedPointerTable =
        (void *)((char *)newKmpTaskDescr + sizeOfTaskInclPrivate);

    newKmpTaskDescr->sub = taskSub;
    newKmpTaskDescr->destructors = NULL;

    PRINT(LD_TASK, "return with task descr kmp: 0x%llx, omptarget-hsail 0x%llx\n",
            P64(newKmpTaskDescr), P64(newExplicitTaskDescr));

    return newKmpTaskDescr;
#endif
    return 0;
}

EXTERN int32_t __kmpc_omp_task_with_deps(
        kmp_Indent *loc,
        uint32_t global_tid,
        kmp_TaskDescr *newKmpTaskDescr,
        int32_t depNum,
        void * depList,
        int32_t noAliasDepNum,
        void * noAliasDepList)
{
    PRINT(LD_IO, "call to __kmpc_omp_task_with_deps(task 0x%llx)\n", P64(newKmpTaskDescr));

#if 0
    // 1. get explict task descr from kmp task descr
    __MSPACE omptarget_hsail_ExplicitTaskDescr *newExplicitTaskDescr = (omptarget_hsail_ExplicitTaskDescr *)
    SUB_BYTES(newKmpTaskDescr, sizeof(omptarget_hsail_TaskDescr));

    ASSERT0(LT_FUSSY, & newExplicitTaskDescr->kmpTaskDescr == newKmpTaskDescr, "bad assumptions");

    __MSPACE omptarget_hsail_TaskDescr *newTaskDescr = & newExplicitTaskDescr->taskDescr;

    ASSERT0(LT_FUSSY, (uint64_t) newTaskDescr == (uint64_t) newExplicitTaskDescr, "bad assumptions");

    // 2. push new context: update new task descriptor
    int gtid = GetGlobalThreadId();
    __MSPACE omptarget_hsail_TaskDescr *parentTaskDescr = getMyTopTaskDescriptor(gtid);
    newTaskDescr->CopyForExplicitTask(parentTaskDescr);
    // set new task descriptor as top
    omptarget_hsail_threadPrivateContext.SetTopLevelTaskDescr(gtid, newTaskDescr);

    // 3. call sub
    PRINT(LD_TASK, "call task sub 0x%llx(task descr 0x%llx)\n", P64(newKmpTaskDescr->sub), P64(newKmpTaskDescr));
    newKmpTaskDescr->sub(0, newKmpTaskDescr);
    PRINT(LD_TASK, "return from call task sub 0x%llx()\n", P64(newKmpTaskDescr->sub));

    // 4. pop context
    omptarget_hsail_threadPrivateContext.SetTopLevelTaskDescr(gtid, parentTaskDescr);

    // 5. free
    SafeFree(newExplicitTaskDescr, "explicit task descriptor");

    return 0;
#endif

    return 0;
}

EXTERN void __kmpc_omp_task_begin_if0(kmp_Indent *loc,
        uint32_t global_tid, kmp_TaskDescr *newKmpTaskDescr)
{
    PRINT(LD_IO, "call to __kmpc_omp_task_begin_if0(task 0x%llx)\n",
    P64(newKmpTaskDescr));
#if 0
    // 1. get explict task descr from kmp task descr
    omptarget_hsail_ExplicitTaskDescr *newExplicitTaskDescr = (omptarget_hsail_ExplicitTaskDescr *);
    SUB_BYTES(newKmpTaskDescr, sizeof(omptarget_hsail_TaskDescr));
    ASSERT0(LT_FUSSY, & newExplicitTaskDescr->kmpTaskDescr == newKmpTaskDescr,
            "bad assumptions");
    omptarget_hsail_TaskDescr *newTaskDescr = & newExplicitTaskDescr->taskDescr;
    ASSERT0(LT_FUSSY, (uint64_t) newTaskDescr == (uint64_t) newExplicitTaskDescr,
            "bad assumptions");

    // 2. push new context: update new task descriptor
    int gtid = GetGlobalThreadId();
    omptarget_hsail_TaskDescr *parentTaskDescr = getMyTopTaskDescriptor(gtid);
    newTaskDescr->CopyForExplicitTask(parentTaskDescr);
    // set new task descriptor as top
    omptarget_hsail_threadPrivateContext.SetTopLevelTaskDescr(gtid, newTaskDescr);
    // 3... noting to call... is inline
    // 4 & 5 ... done in complete
#endif
}

EXTERN void __kmpc_omp_task_complete_if0(kmp_Indent *loc,
        uint32_t global_tid, kmp_TaskDescr *newKmpTaskDescr)
{
    PRINT(LD_IO, "call to __kmpc_omp_task_complete_if0(task 0x%llx)\n",
            P64(newKmpTaskDescr));
#if 0
    // 1. get explict task descr from kmp task descr
    omptarget_hsail_ExplicitTaskDescr *newExplicitTaskDescr = (omptarget_hsail_ExplicitTaskDescr *);
    SUB_BYTES(newKmpTaskDescr, sizeof(omptarget_hsail_TaskDescr));
    ASSERT0(LT_FUSSY, & newExplicitTaskDescr->kmpTaskDescr == newKmpTaskDescr, 
            "bad assumptions");
    omptarget_hsail_TaskDescr *newTaskDescr = & newExplicitTaskDescr->taskDescr; 
    ASSERT0(LT_FUSSY, (uint64_t) newTaskDescr == (uint64_t) newExplicitTaskDescr, 
            "bad assumptions");
    // 2. get parent
    omptarget_hsail_TaskDescr *parentTaskDescr = newTaskDescr->GetPrevTaskDescr();
    // 3... noting to call... is inline
    // 4. pop context
    int gtid = GetGlobalThreadId();
    omptarget_hsail_threadPrivateContext.SetTopLevelTaskDescr(gtid, parentTaskDescr);
    // 5. free
    SafeFree(newExplicitTaskDescr, "explicit task descriptor");
#endif
}

EXTERN void __kmpc_omp_wait_deps(
        kmp_Indent *loc,
        uint32_t global_tid,
        int32_t depNum,
        void * depList,
        int32_t noAliasDepNum,
        void * noAliasDepList)
{
    PRINT0(LD_IO, "call to __kmpc_omp_wait_deps(..)\n");
    // nothing to do as all our tasks are executed as final
}

EXTERN void __kmpc_taskgroup(
        kmp_Indent *loc,
        uint32_t global_tid)
{
    PRINT0(LD_IO, "call to __kmpc_taskgroup(..)\n");
    // nothing to do as all our tasks are executed as final
}

EXTERN void __kmpc_end_taskgroup(
        kmp_Indent *loc,
        uint32_t global_tid)
{
    PRINT0(LD_IO, "call to __kmpc_end_taskgroup(..)\n");
    // nothing to do as all our tasks are executed as final
}

EXTERN void __kmpc_omp_taskyield(
        kmp_Indent *loc,
        uint32_t global_tid)
{
    PRINT0(LD_IO, "call to __kmpc_taskyield()\n");
    // do nothing
}

EXTERN void __kmpc_omp_taskwait(
        kmp_Indent *loc,
        uint32_t global_tid)
{
    PRINT0(LD_IO, "call to __kmpc_taskwait()\n");
    // nothing to do as all our tasks are executed as final
}

