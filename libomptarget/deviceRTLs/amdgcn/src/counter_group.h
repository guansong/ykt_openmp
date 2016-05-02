//===------ counter_group.h - hsail OpenMP loop scheduling ------- HSA -*-===//
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

#ifndef SRC_COUNTER_GROUP_H_
#define SRC_COUNTER_GROUP_H_

//#include <stdlib.h>
//#include <stdio.h>

// counter group type for synchronizations
#if 0
class omptarget_hsail_CounterGroup
{
  public:
    // getters and setters
    INLINE Counter & Event () { return v_event; }
    INLINE volatile Counter & Start () { return v_start; }
    INLINE Counter & Init ()  { return v_init; }

    // Synchronization Interface

    INLINE void    Clear(); // first time start=event
    INLINE void    Reset(); // init = first
    INLINE void    Init(Counter & priv); // priv = init
    INLINE Counter Next();  // just counts number of events

    // set priv to n, to be used in later waitOrRelease
    INLINE void Complete(Counter & priv, Counter n);

    // check priv and decide if we have to wait or can free the other warps
    INLINE void Release(Counter priv, Counter current_event_value);
    INLINE void WaitOrRelease(Counter priv, Counter current_event_value);

  private:
    Counter v_event; // counter of events (atomic)

    // volatile is needed to force loads to read from global
    // memory or L2 cache and see the write by the last master
    volatile Counter v_start; // signal when events registered are finished

    Counter v_init;  // used to initialize local thread variables
};
#endif


typedef struct omptarget_hsail_CounterGroup {
  Counter v_event;
  Counter v_start;
  Counter v_init;
} omptarget_hsail_CounterGroup;


#endif /* SRC_COUNTER_GROUP_H_ */
