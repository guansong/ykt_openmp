//===------- loopi.h - HSAIL OpenMP loop chunking/scheduling support ---- HSA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Modified from Intel CPU and Nvidia GPU for HSAIL
// bitbucket: gansong (zhang.guansong@gmail.com)
//
//===----------------------------------------------------------------------===//

// Generic implementation of OMP loop scheduling with static policy
/*! \brief Calculate initial bounds for static loop and stride
 *  @param[in] loc location in code of the call (not used here)
 *  @param[in] global_tid global thread id
 *  @param[in] schetype type of scheduling (see omptarget-hsail.h)
 *  @param[in] plastiter pointer to last iteration
 *  @param[in,out] pointer to loop lower bound. it will contain value of
 *  lower bound of first chunk
 *  @param[in,out] pointer to loop upper bound. It will contain value of
 *  upper bound of first chunk
 *  @param[in,out] pointer to loop stride. It will contain value of stride
 *  between two successive chunks executed by the same thread
 *  @param[in] loop increment bump
 *  @param[in] chunk size
 */


////////////////////////////////////////////////////////////////////////////////
// Support for dispatch Init

#define MAKE_dispatch_init(x, y) dispatch_init_ ## x ## _ ## y
#define dispatch_init(a, b) MAKE_dispatch_init(a, b)

INLINE static void dispatch_init(LOOPTYPE, CHNKTYPE) (
    kmp_sched_t schedule,
    LOOPTYPE lb,
    LOOPTYPE ub,
    CHNKTYPE st,
    CHNKTYPE chunk)
{
  ASSERT0(LT_FUSSY, lb==0, "exected normalized loop");
  lb = 0;

  int gtid = GetGlobalThreadId();
  omptarget_hsail_TaskDescr * currTaskDescr = getMyTopTaskDescriptorById(gtid);
  LOOPTYPE tnum = currTaskDescr->data.items.threadsInTeam;
  LOOPTYPE tripCount = ub - lb + 1; // +1 because ub is inclusive
  ASSERT0(LT_FUSSY, GetOmpThreadId(gtid)<GetNumberOfOmpThreads(gtid),
      "current thread is not needed here; error");

  // Process schedule.
  if (tnum == 1  || tripCount<=1 || OrderedSchedule(schedule)) {
    PRINT(LD_LOOP,
        "go sequential as tnum=%d, trip count %lld, ordered sched=%d\n",
        tnum, P64(tripCount), schedule);
    schedule = kmp_sched_static_chunk;
    chunk = tripCount; // one thread gets the whole loop

  } else if (schedule == kmp_sched_runtime) {
    // process runtime
    omp_sched_t rtSched = (currTaskDescr->data.items.flags & TaskDescr_SchedMask) +1;
    chunk =  currTaskDescr->data.items.runtimeChunkSize;
    switch (rtSched) {
      case omp_sched_static :
        {
          if (chunk>0) schedule = kmp_sched_static_chunk;
          else schedule = kmp_sched_static_nochunk;
          break;
        }
      case omp_sched_auto :
        {
          schedule = kmp_sched_static_chunk;
          chunk = 1;
          break;
        }
      case omp_sched_dynamic :
      case omp_sched_guided :
        {
          schedule = kmp_sched_dynamic;
          break;
        }
    }
    PRINT(LD_LOOP, "Runtime sched is %d with chunk %lld\n", schedule, P64(chunk));
  } else if (schedule == kmp_sched_auto) {
    schedule = kmp_sched_static_chunk;
    chunk = 1;
    PRINT(LD_LOOP, "Auto sched is %d with chunk %lld\n", schedule, P64(chunk));
  } else {
    PRINT(LD_LOOP, "Dyn sched is %d with chunk %lld\n", schedule, P64(chunk));
    ASSERT(LT_FUSSY, schedule == kmp_sched_dynamic || schedule == kmp_sched_guided,
        "unknown schedule %d & chunk %lld\n",
        schedule, P64(chunk));
  }

  // save sched state
  omptarget_hsail_threadPrivateContext.schedule[gtid] = schedule;
  omptarget_hsail_threadPrivateContext.loopUpperBound[gtid] = ub;

  // init schedules
  if (schedule == kmp_sched_static_chunk) {
    ASSERT0(LT_FUSSY, chunk>0, "bad chunk value");
    // save ub
    omptarget_hsail_threadPrivateContext.loopUpperBound[gtid] = ub;
    // compute static chunk
    CHNKTYPE stride;
    LOOPTYPE  threadId = GetOmpThreadId(gtid);
    ForStaticChunk(LOOPTYPE, CHNKTYPE)(&lb, &ub, &stride, chunk, threadId, tnum);
    // save computed params
    omptarget_hsail_threadPrivateContext.chunk[gtid] = chunk;
    omptarget_hsail_threadPrivateContext.currEvent_or_nextLowerBound[gtid] = lb;
    omptarget_hsail_threadPrivateContext.eventsNum_or_stride[gtid] = stride;
    PRINT(LD_LOOP,
        "dispatch init (static chunk) : num threads = %d, ub = %lld,"
        "next lower bound = %lld, stride = %lld\n",
        GetNumberOfOmpThreads(gtid),
        omptarget_hsail_threadPrivateContext.loopUpperBound[gtid],
        omptarget_hsail_threadPrivateContext.currEvent_or_nextLowerBound[gtid],
        omptarget_hsail_threadPrivateContext.eventsNum_or_stride[gtid]);

  } else if (schedule == kmp_sched_static_nochunk) {
    ASSERT0(LT_FUSSY, chunk==0, "bad chunk value");
    // save ub
    omptarget_hsail_threadPrivateContext.currEvent_or_nextLowerBound[gtid] = ub;
    // compute static chunk
    CHNKTYPE stride;
    LOOPTYPE  threadId = GetOmpThreadId(gtid);
    ForStaticNoChunk(LOOPTYPE, CHNKTYPE) (&lb, &ub, &stride, &chunk, threadId, tnum);
    // save computed params
    omptarget_hsail_threadPrivateContext.chunk[gtid] = chunk;
    omptarget_hsail_threadPrivateContext.currEvent_or_nextLowerBound[gtid] = lb;
    omptarget_hsail_threadPrivateContext.eventsNum_or_stride[gtid] = stride;
    PRINT(LD_LOOP,
        "dispatch init (static nochunk) : num threads = %d, ub = %lld,"
        "next lower bound = %lld, stride = %lld\n",
        GetNumberOfOmpThreads(gtid),
        omptarget_hsail_threadPrivateContext.loopUpperBound[gtid],
        omptarget_hsail_threadPrivateContext.currEvent_or_nextLowerBound[gtid],
        omptarget_hsail_threadPrivateContext.eventsNum_or_stride[gtid]);

  } else if (schedule == kmp_sched_dynamic || schedule == kmp_sched_guided) {
    if (chunk<1) chunk = 1;
    Counter eventNum =  ((tripCount -1) / chunk) + 1; // number of chunks
    // but each thread (but one) must discover that it is last
    eventNum += tnum;
    omptarget_hsail_threadPrivateContext.chunk[gtid] = chunk;
    omptarget_hsail_threadPrivateContext.eventsNum_or_stride[gtid] = eventNum;
    PRINT(LD_LOOP,
        "dispatch init (dyn) : num threads = %d, ub = %lld, chunk %lld, "
        "events number = %lld\n",
        GetNumberOfOmpThreads(gtid),
        omptarget_hsail_threadPrivateContext.loopUpperBound[gtid],
        omptarget_hsail_threadPrivateContext.chunk[gtid],
        omptarget_hsail_threadPrivateContext.eventsNum_or_stride[gtid]);
  }
}


////////////////////////////////////////////////////////////////////////////////
// Support for dispatch next

#define MAKE_DynamicNextChunk(x, y) DynamicNextChunk_ ## x ## _ ## y
#define DynamicNextChunk(a, b) MAKE_DynamicNextChunk(a, b)

INLINE static int DynamicNextChunk(LOOPTYPE, CHNKTYPE) (
    omptarget_hsail_CounterGroup * counters,
    Counter priv,
    LOOPTYPE * lb,
    LOOPTYPE * ub,
    Counter * chunkId,
    Counter * currentEvent,
    LOOPTYPE chunkSize,
    LOOPTYPE loopUpperBound)
{
#if 0
  // get next event atomically
  Counter nextEvent = counters.Next();
  // calculate chunk Id (priv was initialized upon entering the loop to 'start' == 'event')
  chunkId = nextEvent - priv;
  // calculate lower bound for all lanes in the warp
  lb = chunkId * chunkSize;  // this code assume normalization of LB
  ub = lb + chunkSize -1;  // Clang uses i <= ub

  // 3 result cases:
  //  a. lb and ub < loopUpperBound --> NOT_FINISHED
  //  b. lb < loopUpperBound and ub >= loopUpperBound: last chunk --> NOT_FINISHED
  //  c. lb and ub >= loopUpperBound: empty chunk --> FINISHED
  currentEvent = nextEvent;
  // a.
  if (ub <= loopUpperBound) {
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; not finished\n",
        P64(lb), P64(ub), P64(loopUpperBound));
    return NOT_FINISHED;
  }
  // b.
  if (lb <= loopUpperBound) {
    PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; clip to loop ub\n",
        P64(lb), P64(ub), P64(loopUpperBound));
    ub = loopUpperBound;
    return LAST_CHUNK;
  }
  // c. if we are here, we are in case 'c'
  lb = loopUpperBound +1;
  PRINT(LD_LOOPD, "lb %lld, ub %lld, loop ub %lld; finished\n",
      P64(lb), P64(ub), P64(loopUpperBound));
  return FINISHED;
#endif
}

#define MAKE_dispatch_next(x, y) dispatch_next_ ## x ## _ ## y
#define dispatch_next(a, b) MAKE_dispatch_next(a, b)

INLINE static int dispatch_next(LOOPTYPE, CHNKTYPE) (
    int32_t *plast,
    LOOPTYPE *plower,
    LOOPTYPE *pupper,
    CHNKTYPE *pstride)
{
#if 0
  // ID of a thread in its own warp

  // automatically selects thread or warp ID based on selected implementation
  int gtid = GetGlobalThreadId();
  ASSERT0(LT_FUSSY, GetOmpThreadId(gtid)<GetNumberOfOmpThreads(gtid),
      "current thread is not needed here; error");
  // retrieve schedule
  kmp_sched_t schedule = omptarget_hsail_threadPrivateContext.ScheduleType(gtid);

  // xxx reduce to one
  if (schedule == kmp_sched_static_chunk || schedule == kmp_sched_static_nochunk) {
    LOOPTYPE myLb = omptarget_hsail_threadPrivateContext.NextLowerBound(gtid);
    LOOPTYPE ub = omptarget_hsail_threadPrivateContext.LoopUpperBound(gtid);
    // finished?
    if (myLb > ub) {
      PRINT(LD_LOOP, "static loop finished with myLb %lld, ub %lld\n", P64(myLb), P64(ub));
      return DISPATCH_FINISHED;
    }
    // not finished, save current bounds
    CHNKTYPE chunk = omptarget_hsail_threadPrivateContext.Chunk(gtid);
    *plower = myLb;
    LOOPTYPE myUb =  myLb + chunk -1; // Clang uses i <= ub
    if (myUb > ub) myUb = ub;
    *pupper = myUb;

    // increment next lower bound by the stride
    CHNKTYPE stride = omptarget_hsail_threadPrivateContext.Stride(gtid);
    omptarget_hsail_threadPrivateContext.NextLowerBound(gtid) = myLb + stride;
    PRINT(LD_LOOP, "static loop continues with myLb %lld, myUb %lld\n", P64(*plower), P64(*pupper));
    return DISPATCH_NOTFINISHED;
  }
  ASSERT0(LT_FUSSY, schedule==kmp_sched_dynamic || schedule==kmp_sched_guided, "bad sched");
  omptarget_hsail_TeamDescr & teamDescr = getMyTeamDescriptor();
  LOOPTYPE myLb, myUb;
  Counter chunkId;
  // xxx current event is now local
  omptarget_hsail_CounterGroup &counters = teamDescr.WorkDescr().CounterGroup();
  int finished = DynamicNextChunk(counters,
      omptarget_hsail_threadPrivateContext.Priv(gtid), myLb, myUb, chunkId,
      omptarget_hsail_threadPrivateContext.CurrentEvent(gtid),
      omptarget_hsail_threadPrivateContext.Chunk(gtid),
      omptarget_hsail_threadPrivateContext.LoopUpperBound(gtid));

  if (finished == FINISHED) {
    counters.Complete(omptarget_hsail_threadPrivateContext.Priv(gtid),
        omptarget_hsail_threadPrivateContext.EventsNumber(gtid));
    counters.Release (omptarget_hsail_threadPrivateContext.Priv(gtid),
        omptarget_hsail_threadPrivateContext.CurrentEvent(gtid));

    return DISPATCH_FINISHED;
  }

  // not finished (either not finished or last chunk)
  *plower = myLb;
  *pupper = myUb;
  *pstride = 1;

  PRINT(LD_LOOP,"Got sched: active %d, total %d: lb %lld, ub %lld, stride = %lld\n",
      GetNumberOfOmpThreads(gtid), GetNumberOfThreadsInBlock(),
      P64(*plower), P64(*pupper), P64(*pstride));
  return DISPATCH_NOTFINISHED;
#endif

}

