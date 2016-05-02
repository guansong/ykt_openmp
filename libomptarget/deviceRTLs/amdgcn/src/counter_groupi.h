//===----- counter_groupi.h - hsail OpenMP loop scheduling ------- HSA -*-===//
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
//
// Interface implementation for OpenMP loop scheduling
//
//===----------------------------------------------------------------------===//


INLINE void omptarget_hsail_CounterGroup_Clear(omptarget_hsail_CounterGroup * grpPtr)
{
  PRINT0(LD_SYNCD, "clear counters\n")
  grpPtr->v_event = 0;
  grpPtr->v_start = 0;
  // v_init does not need to be reset (its value is dead)
}

INLINE void omptarget_hsail_CounterGroup_Reset(omptarget_hsail_CounterGroup * grpPtr)
{
  // done by master before entering parallel
  ASSERT(LT_FUSSY, grpPtr->v_event==grpPtr->v_start,
    "error, entry %lld !=start %lld at reset\n", P64(grpPtr->v_event), P64(grpPtr->v_start));
  grpPtr->v_init = grpPtr->v_start;
}

INLINE void omptarget_hsail_CounterGroup_Init(omptarget_hsail_CounterGroup * grpPtr, Counter * priv)
{
  PRINT(LD_SYNCD, "init priv counter 0x%llx with val %lld\n",
    P64(&priv), P64(grpPtr->v_start));
  *priv = grpPtr->v_start;
}

// just counts number of events
INLINE Counter omptarget_hsail_CounterGroup_Next(omptarget_hsail_CounterGroup * grpPtr)
{
#if 0
  Counter oldVal = atomicAdd(&v_event, (Counter) 1);
  PRINT(LD_SYNCD, "next event counter 0x%llx with val %lld->%lld\n", 
    P64(&v_event), P64(oldVal), P64(oldVal+1));

  return oldVal;
#endif
  return 0;
}

#if 0
//set priv to n, to be used in later waitOrRelease
INLINE void  omptarget_hsail_CounterGroup::Complete(Counter & priv, Counter n)
{
  PRINT(LD_SYNCD, "complete priv counter 0x%llx with val %lld->%lld (+%d)\n", 
    P64(&priv), P64(priv), P64(priv+n), n);
  priv += n;
}

INLINE void  omptarget_hsail_CounterGroup::Release (
  Counter priv,
  Counter current_event_value)
{
  if (priv - 1 == current_event_value) {
    PRINT(LD_SYNCD, "Release start counter 0x%llx with val %lld->%lld\n", 
      P64(&v_start), P64(v_start), P64(priv));
    v_start = priv;
  } 
}

// check priv and decide if we have to wait or can free the other warps
INLINE void  omptarget_hsail_CounterGroup::WaitOrRelease (
  Counter priv,
  Counter current_event_value)
{
  if (priv - 1 == current_event_value) {
    PRINT(LD_SYNCD, "Release start counter 0x%llx with val %lld->%lld\n", 
      P64(&v_start), P64(v_start), P64(priv));
    v_start = priv;
  } else {
    PRINT(LD_SYNCD, "Start waiting while start counter 0x%llx with val %lld < %lld\n", 
      P64(&v_start), P64(v_start), P64(priv));
    while (priv > v_start) {
      // IDLE LOOP
      // start is volatile: it will be re-loaded at each while loop
    }
    PRINT(LD_SYNCD, "Done waiting as start counter 0x%llx with val %lld >= %lld\n", 
      P64(&v_start), P64(v_start), P64(priv));
  }
}

#endif
