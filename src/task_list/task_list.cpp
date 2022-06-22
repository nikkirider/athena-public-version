//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file task_list.cpp
//  \brief functions for TaskList base class

// needed for vector of pointers in DoTaskListOneStage()
#include <vector>

// Athena++ classes headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"

// this class header
#include "task_list.hpp"

#define DEBUG_ALL
//----------------------------------------------------------------------------------------
// TaskList constructor

TaskList::TaskList(Mesh *pm) {
  pmy_mesh_=pm;
  ntasks = 0;
  nstages = 0;
}

// destructor

TaskList::~TaskList() {
}

//----------------------------------------------------------------------------------------
//! \fn enum TaskListStatus TaskList::DoAllAvailableTasks
//  \brief do all tasks that can be done (are not waiting for a dependency to be
//  cleared) in this TaskList, return status.

enum TaskListStatus TaskList::DoAllAvailableTasks(MeshBlock *pmb, int stage,
                                                  TaskState &ts) {
  int skip=0;
  enum TaskStatus ret;

  if (ts.num_tasks_left==0) return TL_NOTHING_TO_DO;

#ifdef DEBUG_ALL
  fprintf(stdout,"ts.indx_first_task=%4i, ntasks=%4i\n",ts.indx_first_task,ntasks);
#endif

  for (int i=ts.indx_first_task; i<ntasks; i++) {
    Task &taski=task_list_[i];

#ifdef DEBUG_ALL
		fprintf(stdout,"i=%4i,task_id=%4i,tasks_left=%4i,ntasks=%4i\n",i,taski.task_id,ts.num_tasks_left,ntasks);
#endif

    if ((taski.task_id & ts.finished_tasks) == 0LL) { // task not done
      // check if dependency clear
      if (((taski.dependency & ts.finished_tasks) == taski.dependency)) {
#ifdef DEBUG_ALL
        fprintf(stdout,"dependency\n");
#endif
        ret=(this->*task_list_[i].TaskFunc)(pmb, stage);
        if (ret!=TASK_FAIL) { // success
#ifdef DEBUG_ALL
          fprintf(stdout,"success, tasks_left=%4i\n",ts.num_tasks_left);
          fprintf(stdout,"i=%4i,ntasks=%4i\n",i,ntasks);
#endif
          ts.num_tasks_left--;
          ts.finished_tasks |= taski.task_id;
          if (skip==0){
#ifdef DEBUG_ALL
            fprintf(stdout,"skip is zero, ts.indx_first_task=%4i\n",ts.indx_first_task);
#endif
            ts.indx_first_task++;
          }
          if (ts.num_tasks_left==0){
#ifdef DEBUG_ALL
            fprintf(stdout,"num tasks left is zero, tl complete\n");
            fprintf(stdout,"i=%4i,ntasks=%4i\n",i,ntasks);
#endif
            return TL_COMPLETE;
          }
          if (ret==TASK_NEXT){
#ifdef DEBUG_ALL
            fprintf(stdout,"ret is task_next\n");
            fprintf(stdout,"i=%4i,ntasks=%4i\n",i,ntasks);
#endif
            continue;
          }
#ifdef DEBUG_ALL
          fprintf(stdout,"still running\n");
#endif
          return TL_RUNNING;
        }

      }
#ifdef DEBUG_ALL
      fprintf(stdout,"task not done, skip=%4i\n",skip);
#endif
      skip++; // increment number of tasks processed

    } else if (skip==0) { // this task is already done AND it is at the top of the list
      ts.indx_first_task++;
#ifdef DEBUG_ALL
      fprintf(stdout,"task done\n");
#endif
    }
  }
  return TL_STUCK; // there are still tasks to do but nothing can be done now
}

//----------------------------------------------------------------------------------------
//! \fn void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage)
//  \brief completes all tasks in this list, will not return until all are tasks done

void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage) {
  MeshBlock *pmb = pmesh->pblock;
  // initialize counters stored in each MeshBlock
  while (pmb != NULL)  {
    pmb->tasks.Reset(ntasks);
    pmb=pmb->next;
  }

  // initialize a vector of MeshBlock pointers
  int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock*> pmb_array(nmb);
  pmb = pmesh->pblock;
  for (int i=0; i<nmb; ++i) {
    pmb_array[i] = pmb;
    pmb=pmb->next;
  }

  int nmb_left = nmb;
  int nthreads = pmesh->GetNumMeshThreads();

  // cycle through all MeshBlocks and perform all tasks possible
  while(nmb_left > 0) {
#ifdef DEBUG_ALL
  fprintf(stdout,"nmb=%4i,nmb_left=%4i\n",nmb,nmb_left);
#endif
#pragma omp parallel shared(nmb_left) num_threads(nthreads)
{
    #pragma omp for reduction(- : nmb_left) schedule(dynamic,1)
    for (int i=0; i<nmb; ++i) {
      if (DoAllAvailableTasks(pmb_array[i], stage, pmb_array[i]->tasks) == TL_COMPLETE) {
#ifdef DEBUG_ALL
        fprintf(stdout,"i=%4i,ntasks=%4i,nmb=%4i\n",i,ntasks,nmb);
#endif
        nmb_left--;
      }
    }
}

  }

  return;
}
