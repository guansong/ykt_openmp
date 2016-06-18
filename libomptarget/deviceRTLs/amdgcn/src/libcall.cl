//===------------ libcall.cl - hsail OpenMP user calls ----------- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP runtime functions that can be
// invoked by the user in an OpenMP region
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

#include "omptarget-hsail.h"


EXTERN double omp_get_wtick(void)
{
    double rc = omptarget_hsail_globalICV.gpuCycleTime;

    PRINT(LD_IO, "call omp_get_wtick() returns %g\n", rc);
    return rc;
}

EXTERN double omp_get_wtime(void)
{
    //FIXME: clock64()?
    double rc = omptarget_hsail_globalICV.gpuCycleTime * 1;//clock64();

    PRINT(LD_IO, "call omp_get_wtime() returns %g\n", rc);
    return rc;
}

EXTERN void omp_set_num_threads(int num)
{
    if (num <= 0) {
        WARNING0(LW_INPUT, "expected positive num; ignore\n");
    } else {
        __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
        currTaskDescr->data.items.nthreads = num;
    }

    PRINT(LD_IO, "call omp_set_num_threads(num %d)\n", num);
}

EXTERN int omp_get_num_threads(void)
{
    int gtid = GetGlobalThreadId();
    int rc = GetNumberOfOmpThreads(gtid);

    PRINT(LD_IO, "call omp_get_num_threads() return %d\n", rc);
    return rc;
}

EXTERN int omp_get_max_threads(void)
{
    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    // default is 1 thread avail
    int rc = 1;

    if (! currTaskDescr->data.items.flags & TaskDescr_InPar) {
        // not currently in a parallel region... all are available
        rc = GetNumberOfProcsInTeam();
        ASSERT0(LT_FUSSY, rc >= 0, "bad number of threads");
    }

    PRINT(LD_IO, "call omp_get_num_threads() return %\n", rc);
    return rc;
}

EXTERN int omp_get_thread_limit(void)
{
    // per contention group.. meaning threads in current team
    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();

    int rc = currTaskDescr->data.items.threadlimit;

    PRINT(LD_IO, "call omp_get_thread_limit() return %d\n", rc);
    return rc;
}

EXTERN int omp_get_thread_num()
{
    int gtid = GetGlobalThreadId();

    int rc = GetOmpThreadId(gtid);

    PRINT(LD_IO, "call omp_get_thread_num() returns %d\n", rc);
    return rc;
}

EXTERN int omp_get_num_procs(void)
{
    int rc = GetNumberOfThreadsInBlock();

    PRINT(LD_IO, "call omp_get_num_procs() returns %d\n", rc);
    return rc;
}

EXTERN int omp_in_parallel(void)
{
    int rc = 0;

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    if (currTaskDescr->data.items.flags & TaskDescr_InPar) {
        rc = 1;
    }

    PRINT(LD_IO, "call omp_in_parallel() returns %d\n", rc);
    return rc;
}

EXTERN int omp_in_final(void)
{
    // treat all tasks as final... Specs may expect runtime to keep
    // track more precisely if a task was actively set by users... This
    // is not explicitely specified; will treat as if runtime can
    // actively decide to put a non-final task into a final one.
    int rc = 1;

    PRINT(LD_IO, "call omp_in_final() returns %d\n", rc);
    return rc;
}

EXTERN void omp_set_dynamic(int flag)
{
    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();

    if (flag) {
        currTaskDescr->data.items.flags |= TaskDescr_IsDynamic;
    } else {
        currTaskDescr->data.items.flags &= (~TaskDescr_IsDynamic);
    }

    PRINT(LD_IO, "call omp_set_dynamic(%d)\n", flag);
}

EXTERN int  omp_get_dynamic(void)
{
    int rc = 0;

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    if (currTaskDescr->data.items.flags & TaskDescr_IsDynamic) {
        rc = 1;
    }

    PRINT(LD_IO, "call omp_get_dynamic() returns %d\n", rc);
    return rc;
}

EXTERN void omp_set_nested(int flag)
{
    PRINT(LD_IO, "call omp_set_nested(%d) is ignored (no nested support)\n", flag);
}

EXTERN int omp_get_nested(void)
{
    int  rc = 0;

    PRINT(LD_IO, "call omp_get_nested() returns %d\n", rc);
    return rc;
}

EXTERN void omp_set_max_active_levels(int level)
{
    PRINT(LD_IO, "call omp_set_max_active_levels(%d) is ignored (no nested support)\n", level);
}

EXTERN int omp_get_max_active_levels(void)
{
    int  rc = 1;

    PRINT(LD_IO, "call omp_get_nested() returns %d\n", rc);
    return rc;
}

EXTERN int omp_get_level(void)
{
    int level = 0;

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");

    do {
        if (currTaskDescr->data.items.flags & TaskDescr_IsParConstr) {
            level++;
        }
        currTaskDescr = currTaskDescr->prev;
    } while (currTaskDescr);

    PRINT(LD_IO, "call omp_get_level() returns %d\n", level);
    return level;
}

EXTERN int omp_get_active_level(void)
{
    int level = 0; // no active level parallelism

    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");

    do {
        if (currTaskDescr->data.items.threadsInTeam > 1) {
            // has a parallel with more than one thread in team
            level = 1;
            break;
        }
        currTaskDescr = currTaskDescr->prev;
    } while (currTaskDescr);

    PRINT(LD_IO, "call omp_get_active_level() returns %d\n", level);
    return level;
}

EXTERN int omp_get_ancestor_thread_num(int level)
{
    // default at level 0
    int rc = 0;

    if (level>=0) {
        int totLevel = omp_get_level();
        if (level<=totLevel) {
            __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
            int steps = totLevel - level;

            //PRINT(LD_IO, "backtrack %d steps\n", steps);
            ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");

            do {
                //if (DON(LD_IOD)) PrintTaskDescr(currTaskDescr, (char *)"ancestor", steps);
                if (currTaskDescr->data.items.flags & TaskDescr_IsParConstr) {
                    // found the level
                    if (! steps) {
                        rc = currTaskDescr->data.items.threadId;
                        break;
                    }
                    steps--;
                }
                currTaskDescr = currTaskDescr->prev;
            } while (currTaskDescr);

            ASSERT0(LT_FUSSY, ! steps, "expected to find all steps");
        }
    }

    PRINT(LD_IO, "call omp_get_ancestor_thread_num(level %d) returns %d\n", level, rc);
    return rc;
}


EXTERN int omp_get_team_size(int level)
{
    // default at level 0
    int rc = 1;

    if (level>=0) {
        int totLevel = omp_get_level();

        if (level<=totLevel) {
            __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
            int steps = totLevel - level;

            ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");

            do {
                if (currTaskDescr->data.items.flags & TaskDescr_IsParConstr) {
                    if (! steps) {
                        // found the level
                        rc = currTaskDescr->data.items.threadsInTeam;
                        break;
                    }
                    steps--;
                }
                currTaskDescr = currTaskDescr->prev;
            } while (currTaskDescr);

            ASSERT0(LT_FUSSY, ! steps, "expected to find all steps");
        }
    }

    PRINT(LD_IO, "call omp_get_team_size(level %d) returns %d\n", level, rc);
    return rc;
}


EXTERN void omp_get_schedule(omp_sched_t * kind, int * modifier)
{
    __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();

    // sched starts from 1..4; encode it as 0..3; so add 1 here
    *kind = (currTaskDescr->data.items.flags & TaskDescr_SchedMask) +1;
    *modifier = currTaskDescr->data.items.runtimeChunkSize;

    PRINT(LD_IO, "call omp_get_schedule returns sched %d and modif %d\n", (int) *kind, *modifier);
}

EXTERN void omp_set_schedule(omp_sched_t kind, int modifier)
{
    if (kind>=omp_sched_static && kind<omp_sched_auto) {
        __MSPACE omptarget_hsail_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();

        //currTaskDescr->SetRuntimeSched(kind);
        // sched starts from 1..4; encode it as 0..3; so add 1 here
        uint8_t val = ((uint8_t) kind) -1;
        // clear current sched
        currTaskDescr->data.items.flags &= ~TaskDescr_SchedMask;
        // set new sched
        currTaskDescr->data.items.flags |= val; 

        currTaskDescr->data.items.runtimeChunkSize = modifier;

        PRINT(LD_IOD, "omp_set_schedule did set sched %d & modif %d\n",
                (int) (currTaskDescr->data.items.flags & TaskDescr_SchedMask) + 1,
                currTaskDescr->data.items.runtimeChunkSize);
    } else {
        PRINT(LD_IO, "call omp_set_schedule(sched %d, modif %d) ignored\n", (int) kind, modifier);
    }
}

EXTERN omp_proc_bind_t omp_get_proc_bind(void)
{
    PRINT0(LD_IO, "call omp_get_proc_bin() is true, regardless on state\n");
    return omp_proc_bind_true;
}

EXTERN int  omp_get_cancellation(void)
{
    int rc = omptarget_hsail_globalICV.cancelPolicy;

    PRINT(LD_IO, "call omp_get_cancellation() returns %d\n", rc);
    return rc;
}

EXTERN void omp_set_default_device(int deviceId)
{
    PRINT0(LD_IO, "call omp_get_default_device() is undef on device\n");
}

EXTERN int  omp_get_default_device(void)
{
    PRINT0(LD_IO, "call omp_get_default_device() is undef on device, returns 0\n");
    return 0;
}

EXTERN int  omp_get_num_devices(void)
{
    PRINT0(LD_IO, "call omp_get_num_devices() is undef on device, returns 0\n");
    return 0;
}

EXTERN int  omp_get_num_teams(void)
{
    int rc = GetNumberOfOmpTeams();

    PRINT(LD_IO, "call omp_get_num_teams() returns %d\n", rc);
    return rc;
}

EXTERN int omp_get_team_num()
{
    int rc = GetOmpTeamId();

    PRINT(LD_IO, "call omp_get_team_num() returns %d\n", rc);
    return rc;
}

EXTERN int omp_is_initial_device(void)
{
    PRINT0(LD_IO, "call omp_is_initial_device() returns 0\n");
    // 0 by def on device
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// locks
////////////////////////////////////////////////////////////////////////////////
// http://developer.amd.com/community/blog/2015/01/15/opencl-2-0-fine-grain-shared-virtual-memory/

#define __OMP_SPIN 1000
#define UNSET 0
#define SET 1


#if OMPTARGET_HSAIL_DEBUG == 0
// Do not use OpenCL builtin functions
EXTERN int __compare_exchange_int_global(__global omp_lock_t * lock, int * comp, int val)
{
    // implemented in hsa_math.bc
    EXTERN int __hsail_atomic_compare_exchange_int_global(__global omp_lock_t *, int, int);
    return __hsail_atomic_compare_exchange_int_global(lock, *comp, val);
}

EXTERN int __exchange_int_global(__global omp_lock_t * lock, int val)
{
    // implemented in hsa_math.bc
    EXTERN int __hsail_atomic_exchange_int_global(__global omp_lock_t *, int);
    return __hsail_atomic_exchange_int_global(lock, val);
}

#else
// Use the OpenCL builtin functions
#endif

EXTERN void omp_init_lock(__global omp_lock_t * lock)
{
    atomic_store(lock, UNSET);
    PRINT0(LD_IO, "call omp_init_lock()\n");
}

EXTERN void omp_destroy_lock(__global omp_lock_t * lock)
{
    PRINT0(LD_IO, "call omp_destroy_lock()\n");
}

EXTERN void omp_set_lock(__global omp_lock_t * lock)
{
    int compare = UNSET;
    int val = SET;

    int ret = -1;
    // FIXME how to use volatile???
    //while (!atomic_compare_exchange_strong((volatile __global omp_lock_t *) lock, &compare, val)) {
    while (!atomic_compare_exchange_strong((__global omp_lock_t *)lock, &compare, val)) {
        //clock_t start = clock();
        //clock_t now;
        for (;;)
        {
            // TODO: not sure spinning is a good idea here..
#if 0
            now = clock();
            clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
            if (cycles >= __OMP_SPIN*blockIdx.x) {
                break;
            }
#else
            for (int i = 0; i< __OMP_SPIN; i++)
            {
                // do what to ensure the compiler do not optimize this away?
            }
            break;
#endif
        }
    } // wait for 0 to be the read value

    PRINT0(LD_IO, "call omp_set_lock()\n");
}

EXTERN void omp_unset_lock(__global omp_lock_t * lock)
{
    int compare = SET;
    int val = UNSET;
    int ret = -1;

    ret = atomic_compare_exchange_strong((__global omp_lock_t *)lock, &compare, val);

    PRINT(LD_IO, "call omp_unset_lock(), rec %d\n", ret);
}

EXTERN int omp_test_lock(__global omp_lock_t * lock)
{
    int compare = UNSET;
    int val = SET;
    int ret = -1;

    // TODO: should check for the lock to be SET?
    ret = atomic_compare_exchange_strong((__global omp_lock_t *)lock, &compare, val);

    PRINT(LD_IO, "call omp_test_lock() return %d\n", ret);
    return ret;
}

#if 0
EXTERN void omp_init_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_destroy_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_set_nest_lock(omp_nest_lock_t *lock);
EXTERN void omp_unset_nest_lock(omp_nest_lock_t *lock);
EXTERN int  omp_test_nest_lock(omp_nest_lock_t *lock);
#endif

