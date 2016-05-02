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
// Loop with static scheduling with chunk
// helper function for static chunk

#define MAKE_ForStaticChunk(x, y) ForStaticChunk_ ## x ## _ ## y
#define ForStaticChunk(a, b) MAKE_ForStaticChunk(a, b)

INLINE static void ForStaticChunk(LOOPTYPE, CHNKTYPE) (
    __private LOOPTYPE *lb,
    __private LOOPTYPE *ub,
    __private CHNKTYPE *stride,
    CHNKTYPE chunk,
    LOOPTYPE entityId,
    LOOPTYPE numberOfEntities)
{
  // each thread executes multiple chunks all of the same size, except
  // the last one
  // distance between two successive chunks
  (*stride) = numberOfEntities * chunk;
  (*lb) = (*lb) + entityId * chunk;
  (*ub) = (*lb) + chunk - 1; // Clang uses i <= ub
}

////////////////////////////////////////////////////////////////////////////////
// Loop with static scheduling without chunk
// helper function for static no chunk

#define MAKE_ForStaticNoChunk(x, y) ForStaticNoChunk_ ## x ## _ ## y
#define ForStaticNoChunk(a, b) MAKE_ForStaticNoChunk(a, b)

INLINE static void ForStaticNoChunk(LOOPTYPE, CHNKTYPE) (
    __private LOOPTYPE *lb,
    __private LOOPTYPE *ub,
    __private CHNKTYPE *stride,
    __private CHNKTYPE *chunk,
    LOOPTYPE entityId,
    LOOPTYPE numberOfEntities)
{
  // No chunk size specified.  Each thread or warp gets at most one
  // chunk; chunks are all almost of equal size
  LOOPTYPE loopSize = (*ub) - (*lb) + 1;

  (*chunk) = loopSize / numberOfEntities;

  LOOPTYPE leftOver = loopSize - (*chunk) * numberOfEntities;

  if (entityId < leftOver) {
    (*chunk)++;
    (*lb) = (*lb) + entityId * (*chunk);
  } else {
    (*lb) = (*lb) + entityId * (*chunk) + leftOver;
  }

  (*ub) = (*lb) + (*chunk) - 1; // Clang uses i <= ub
  (*stride) = loopSize; // make sure we only do 1 chunk per warp
}

////////////////////////////////////////////////////////////////////////////////
// Support for Static Init

#define MAKE_for_static_init(x, y) for_static_init_ ## x ## _ ## y
#define for_static_init(a, b) MAKE_for_static_init(a, b)

INLINE static void for_static_init(LOOPTYPE, CHNKTYPE) (
    int32_t schedtype,
    __private LOOPTYPE *plower,
    __private LOOPTYPE *pupper,
    __private CHNKTYPE *pstride,
    CHNKTYPE chunk)
{
  int gtid = GetGlobalThreadId();

  // Assume we are in teams region or that we use a single block
  // per target region
  CHNKTYPE numberOfActiveOMPThreads = GetNumberOfOmpThreads(gtid);

  // All warps that are in excess of the maximum requested, do
  // not execute the loop
  ASSERT0(LT_FUSSY, GetOmpThreadId(gtid)<GetNumberOfOmpThreads(gtid),
      "current thread is not needed here; error");
  PRINT(LD_LOOP, "OMP Thread %d: schedule type %d, chunk size = %lld\n",
      GetOmpThreadId(gtid), schedtype, P64(chunk));

  // copy
  LOOPTYPE lb = *plower;
  LOOPTYPE ub = *pupper;
  CHNKTYPE stride = *pstride;
  LOOPTYPE entityId, numberOfEntities;
  // init
  switch (schedtype)
  {
    case kmp_sched_static_chunk :
      {
        if (chunk > 0) {
          entityId = GetOmpThreadId(gtid);
          numberOfEntities = GetNumberOfOmpThreads(gtid);
          ForStaticChunk(LOOPTYPE,CHNKTYPE)(&lb, &ub, &stride, chunk, entityId, numberOfEntities);
          break;
        }
      } // note: if chunk <=0, use nochunk
    case kmp_sched_static_nochunk :
      {
        entityId = GetOmpThreadId(gtid);
        numberOfEntities = GetNumberOfOmpThreads(gtid);
        ForStaticNoChunk(LOOPTYPE,CHNKTYPE)(&lb, &ub, &stride, &chunk, entityId, numberOfEntities);
        break;
      }
    case kmp_sched_distr_static_chunk :
      {
        if (chunk > 0) {
          entityId = GetOmpTeamId();
          numberOfEntities = GetNumberOfOmpTeams();
          ForStaticChunk(LOOPTYPE,CHNKTYPE)(&lb, &ub, &stride, chunk, entityId, numberOfEntities);
          break;
        } // note: if chunk <=0, use nochunk
      }
    case kmp_sched_distr_static_nochunk :
      {
        entityId = GetOmpTeamId();
        numberOfEntities = GetNumberOfOmpTeams();

        ForStaticNoChunk(LOOPTYPE,CHNKTYPE)(&lb, &ub, &stride, &chunk, entityId, numberOfEntities);
        break;
      }
    default:
      {
        ASSERT(LT_FUSSY, FALSE, "unknown schedtype %d", schedtype);
        PRINT(LD_LOOP, "unknown schedtype %d, revert back to static chunk\n",
            schedtype);
        entityId = GetOmpThreadId(gtid);
        numberOfEntities = GetNumberOfOmpThreads(gtid);
        ForStaticChunk(LOOPTYPE,CHNKTYPE)(&lb, &ub, &stride, chunk, entityId, numberOfEntities);
      }
  }
  // copy back
  *plower = lb;
  *pupper = ub;
  *pstride = stride;
  PRINT(LD_LOOP,"Got sched: Active %d, total %d: lb %lld, ub %lld, stride %lld\n",
      GetNumberOfOmpThreads(gtid), GetNumberOfThreadsInBlock(),
      P64(*plower), P64(*pupper), P64(*pstride));
}

