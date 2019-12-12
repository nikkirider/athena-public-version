//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file expand.cpp
//  \brief Problem generator for testing expanding grid  
//
//

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>
#include <float.h>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../bvals/bvals.hpp"

//====================================================================================
// global variables

#ifdef MPI_PARALLEL
typedef struct MPI_Comm_Sub {
  MPI_Group gsub, gworld;
  MPI_Comm  comsub;
} MPI_Comm_Sub;

MPI_Comm_Sub comm_x1;
MPI_Comm_Sub comm_slab;
#endif

//====================================================================================
// local functions
void InflowOuterX1_Comoving(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void ReflectInnerX1_nonuniform(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void CMLockToShock(Mesh *pm, AthenaArray<Real> &LockingData, Real dT);//, AthenaArray<Real> &vx2f, AthenaArray<Real> &vx3f);

Real CMMove(AthenaArray<Real> LockData, Real xf, int dir, Real dT, Real time);

Real CMTimeStep(MeshBlock *pmb);

//====================================================================================
// Enroll user-specific functions
void Mesh::InitUserMeshData(ParameterInput *pin) {

  if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
     EnrollUserBoundaryFunction(INNER_X1, ReflectInnerX1_nonuniform);
  }

  if (COMOVING == 1){
    EnrollComovingLockingFunction(CMLockToShock);
    EnrollFaceCoordFunction(CMMove);
    EnrollUserBoundaryFunction(OUTER_X1, InflowOuterX1_Comoving);
    EnrollUserTimeStepFunction(CMTimeStep);
    CMLockData(2)= pin->GetOrAddReal("problem","v0",0.0);
  }
  
  return;
}

//=========================================================================================
//inflow condition -> gets average density and adds it at boundary
void InflowOuterX1_Comoving(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh){

  //Get Average Density here
  Real RhoAve = 0.0; 

  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IVX,k,j,is-i) = -prim(IVX,k,j,(is+i-1));  // reflect 1-velocity
          }
        }
      }
    } else if (n==(IDN)){  
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,is-i) = RhoAve+ prim(n,k,j,(is+i-1));
          }
        }
      }
     
    } else {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,is-i) = prim(n,k,j,(is+i-1));
          }
        }
      }

    }
  }
  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = -b.x1f(k,j,(is+i  ));  // reflect 1-field
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) =  b.x2f(k,j,(is+i-1));
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) =  b.x3f(k,j,(is+i-1));
        }
      }
    }
  }
  return;
  //
   
}



// ======= =================================================================================
// Set data necessary for expanding the grid.
void CMLockToShock(Mesh *pm, AthenaArray<Real> &LockingData, Real dT){ //AthenaArray<Real> &vx2f, AthenaArray<Real> &vx3f){
  RegionSize meshDim = pm->mesh_size;
  LockingData(0) = meshDim.x1min;
  LockingData(1) = (meshDim.x1max) * (93.0/100.0);   
  //LockingData(2) = dT;
  //LockingData(3) = 2.0*(pm->time);
  //std::cout << "Here is LockData(1) in CMLockToShock: " << LockingData(1) << std::endl;
  //std::cout << "Here is LockData(0) in CMLockToShock: " << LockingData(0) << std::endl;
  //std::cout << meshDim.x1max << std::endl;
  return;
  //std::cout << LockingData(1) << std::endl;
}

//=========================================================================================
// Take new comoving data and previous grid cell position and returning new cell position
// Mesh is accessible here, as thisfunction gets called by meshblock
Real CMMove(AthenaArray<Real> LockData, Real xf, int dir, Real dT, Real time){
  
  Real retval;
  if (dir != 0){
    retval = 0.0;
  } else if (xf==LockData(0)){
    retval = 0.0;
  } else if (LockData(1) == LockData(0)){
    retval = 0.0;
  } else {
    Real x0 = LockData(0);
    Real alpha = LockData(1);
    //std::cout << "alpha " << alpha << std::endl;
    Real delAlpha = -1.0*(dT)*(LockData(2))*std::sin(time*3.0);
    //if in x-direction
    retval = (xf-x0) / (alpha-x0) * delAlpha;
    //std::cout << Vel  <<std::endl;
    //std::cout << dT << std::endl;
  } 
  //std::cout << retval << std::endl; 
  return retval;
}
//=========================================================================================
//Make sure grid does not move too far
Real CMTimeStep(MeshBlock *pmb){ 
  Real min_dt = FLT_MAX;
  Real nextPosDelta, minCellSize, dt;
  
  Mesh *pmesh = pmb->pmy_mesh;
  Real Nx1 = pmesh->mesh_size.nx1;
  Real Nx2 = pmesh->mesh_size.nx2;
  Real Nx3 = pmesh->mesh_size.nx3; 
  
  for (int i = 0; i<=Nx1+1;i++){
    nextPosDelta = pmesh->CMNewCoord_(pmesh->CMLockData,pmb->pcoord->x1f(i),0, pmesh->dt, pmesh->time);
    //std::cout << i <<  "th NextPosDelta " << nextPosDelta  << std::endl;
    if (nextPosDelta < 0 && i != 0){ 
      minCellSize = pmb->pcoord->dx1f(i-1);
    } else {
      minCellSize = pmb->pcoord->dx1f(i);
    }
    //minCellSize = std::min(pmb->pcoord->dx1f(i),pmb->pcoord->dx1f(i-1));
    if (nextPosDelta != 0.0 && minCellSize != 0.0){
      dt = std::abs(minCellSize* 0.5/nextPosDelta * (pmesh->dt));
      min_dt = std::min(min_dt, dt);
    } else {
      dt = FLT_MAX;
      min_dt = std::min(min_dt,dt);
    }

  }
  if (Nx2 > 1){
    for (int j = 0; j<=Nx2+1;j++){
      nextPosDelta = pmesh->CMNewCoord_(pmesh->CMLockData,pmb->pcoord->x2f(j),1, pmesh->dt, pmesh->time);
      minCellSize = std::min(pmb->pcoord->dx2f(j),pmb->pcoord->dx2f(j-1));
      dt = minCellSize* 0.5/nextPosDelta * (pmesh->dt);
      min_dt = std::min(min_dt, dt);      
      if (nextPosDelta != 0.0){
        dt = minCellSize* 0.5/nextPosDelta * (pmesh->dt);
        min_dt = std::min(min_dt, dt);
      }   
    }
  }
  if (Nx3 >1) {
    for (int k = 0; k<=Nx3+1;k++){
      nextPosDelta = pmesh->CMNewCoord_(pmesh->CMLockData,pmb->pcoord->x3f(k),2, pmesh->dt,pmesh->time);
      minCellSize = std::min(pmb->pcoord->dx3f(k),pmb->pcoord->dx3f(k-1));
      dt = minCellSize* 0.5/nextPosDelta * (pmesh->dt);
      min_dt = std::min(min_dt, dt);
      if (nextPosDelta != 0.0){
        dt = minCellSize* 0.5/nextPosDelta * (pmesh->dt);
        min_dt = std::min(min_dt, dt);
      }
    }
  }

  //std::cout << "This is my Min dt from grid growth: " << min_dt << std::endl;
  return min_dt;



}



//========================================================================================
// Reflecting inner X1 boundary conditions for radially non-uniform grids

void ReflectInnerX1_nonuniform(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IVX,k,j,is-i) = -prim(IVX,k,j,(is+i-1));  // reflect 1-velocity
          }
        }
      }
    } else {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,is-i) = prim(n,k,j,(is+i-1));
          }
        }
      }
    }
  }
  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = -b.x1f(k,j,(is+i  ));  // reflect 1-field
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) =  b.x2f(k,j,(is+i-1));
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) =  b.x3f(k,j,(is+i-1));
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::InitOTFOutput(ParameterInput *pin)
//  \brief Sets data structures etc for on-the-fly analysis.
//========================================================================================

void MeshBlock::InitOTFOutput(ParameterInput *pin) {
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
 
  Real pi         = 4.0*std::atan(1.0);
  //Real r0         = pin->GetReal("problem","r0"); // radius of initial ejecta
  //Real dr         = pin->GetReal("problem","dr"); // width of transition
  Real p0         = pin->GetReal("problem","p0"); // ambient pressure
  Real d0         = pin->GetReal("problem","d0"); // ambient density
  //Real v0         = pin->GetReal("problem","v0"); // velocity of ejecta
  Real m0         = pin->GetReal("problem","m0"); // mass of ejecta
  Real E0         = pin->GetReal("problem","E0"); // total energy of ejecta
  Real x1min      = pin->GetReal("mesh","x1min");

  if (MAGNETIC_FIELDS_ENABLED) {
    Real b0    = pin->GetReal("problem","b0");
    Real angle = (PI/180.0)*pin->GetReal("problem","angle");
  } else {
    Real b0 = 0.0;
    Real angle = 0.0;
  }
  Real gamma   = peos->GetGamma();
  Real gm1     = gamma - 1.0;
  Real T       = gm1*E0/m0; 
  Real rho0    = d0;
  Real e0      = p0/gm1;
  
   // get coordinates of center of blast, and convert to Cartesian if necessary
  //Real x1_0   = pin->GetOrAddReal("problem","x1_0",0.0);
  //Real x2_0   = pin->GetOrAddReal("problem","x2_0",0.0);
  //Real x3_0   = pin->GetOrAddReal("problem","x3_0",0.0);
  //Real x0,y0,z0;
  //if (COORDINATE_SYSTEM == "cartesian") {
  //  x0 = x1_0;
  //  y0 = x2_0;
  //  z0 = x3_0;
  //} else if (COORDINATE_SYSTEM == "cylindrical") {
  //  x0 = x1_0*std::cos(x2_0);
  //  y0 = x1_0*std::sin(x2_0);
  //  z0 = x3_0;
  //} else if (COORDINATE_SYSTEM == "spherical_polar") {
  //  x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
  //  y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
  //  z0 = x1_0*std::cos(x2_0);
  //} else {
  //  // Only check legality of COORDINATE_SYSTEM once in this function
  //  std::stringstream msg;
  //  msg << "### FATAL ERROR in knovae.cpp ProblemGenerator" << std::endl
  //      << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
  //  throw std::runtime_error(msg.str().c_str());
  //}

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
       // Real rad,thet;
       // if (COORDINATE_SYSTEM == "cartesian") {
       //   Real x   = pcoord->x1v(i);
       //   Real y   = pcoord->x2v(j);
       //   Real z   = pcoord->x3v(k);
       //   rad      = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
       //   thet   = std::acos(z/rad);
       // } else if (COORDINATE_SYSTEM == "cylindrical") {
       //   Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
       //   Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
       //   Real z = pcoord->x3v(k);
       //   rad    = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
       //   thet   = std::acos(z/rad);
       // } else { // if (COORDINATE_SYSTEM == "spherical_polar")
       //   Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
       //   Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
       //   Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
       //   rad    = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
       //   thet   = pcoord->x2v(j);
       // }

        phydro->u(IDN,k,j,i) = rho0; 
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = e0+0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
            phydro->u(IEN,k,j,i) += phydro->u(IDN,k,j,i);
        }
        if (DUAL_ENERGY) {
          phydro->u(IIE,k,j,i) = e0;
        }
        
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    //for (int k = ks; k <= ke; ++k) {
    //  for (int j = js; j <= je; ++j) {
    //    for (int i = is; i <= ie+1; ++i) {
    //      if (COORDINATE_SYSTEM == "cartesian") {
    //        pfield->b.x1f(k,j,i) = b0 * std::cos(angle);
    //      } else if (COORDINATE_SYSTEM == "cylindrical") {
    //        Real phi = pcoord->x2v(j);
    //        pfield->b.x1f(k,j,i) = b0 * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
    //      } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
    //        Real theta = pcoord->x2v(j);
    //        Real phi = pcoord->x3v(k);
    //        pfield->b.x1f(k,j,i) = b0 * std::abs(std::sin(theta))
    //            * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
    //      }
    //    }
    //  }
    //}
    //for (int k = ks; k <= ke; ++k) {
    //  for (int j = js; j <= je+1; ++j) {
    //    for (int i = is; i <= ie; ++i) {
    //      if (COORDINATE_SYSTEM == "cartesian") {
    //        pfield->b.x2f(k,j,i) = b0 * std::sin(angle);
    //      } else if (COORDINATE_SYSTEM == "cylindrical") {
    //        Real phi = pcoord->x2v(j);
    //        pfield->b.x2f(k,j,i) =
    //            b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
    //      } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
    //        Real theta = pcoord->x2v(j);
    //        Real phi = pcoord->x3v(k);
    //        pfield->b.x2f(k,j,i) = b0 * std::cos(theta)
    //            * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
    //        if (std::sin(theta) < 0.0)
    //          pfield->b.x2f(k,j,i) *= -1.0;
    //      }
    //    }
    //  }
    //}
    //for (int k = ks; k <= ke+1; ++k) {
    //  for (int j = js; j <= je; ++j) {
    //    for (int i = is; i <= ie; ++i) {
    //      if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
    //        pfield->b.x3f(k,j,i) = 0.0;
    //      } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
    //        Real phi = pcoord->x3v(k);
    //        pfield->b.x3f(k,j,i) =
    //            b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
    //      }
    //    }
    //  }
    //}
    //for (int k = ks; k <= ke; ++k) {
    //  for (int j = js; j <= je; ++j) {
    //    for (int i = is; i <= ie; ++i) {
    //      phydro->u(IEN,k,j,i) += 0.5*b0*b0;
    //    }
    //  }
    //}
  }

#ifdef WURSTBROT  
  if (lid == 0) { // only defined on root level
    // count ranks in local loc.lx2,loc.lx3 column. Complicated way to express nrbx1.
    int nbx1count=0;
    for (int nbt=0; nbt<pmy_mesh->nbtotal; nbt++) {
      if (  (loc.lx2   == pmy_mesh->loclist[nbt].lx2)
          &&(loc.lx3   == pmy_mesh->loclist[nbt].lx3)
          &&(loc.level == pmy_mesh->loclist[nbt].level)) {
        nbx1count++;
      }
    }
    int* bx1ranks = new int[nbx1count]; // rank indices along one column (duplicates possible)
    int* bx1block = new int[nbx1count]; // block indices along one column (no duplicates)
    // again, now we assign the ranks
    int ibx1count = 0; // use as index
    for (int nbt=0; nbt<pmy_mesh->nbtotal; nbt++) {
      if (  (loc.lx2   == pmy_mesh->loclist[nbt].lx2)
          &&(loc.lx3   == pmy_mesh->loclist[nbt].lx3)
          &&(loc.level == pmy_mesh->loclist[nbt].level)) {
        bx1ranks[ibx1count] = pmy_mesh->ranklist[nbt];
        bx1block[ibx1count] = nbt;
        ibx1count++;
      }
    }
    //for (int nbt=0; nbt<nbx1count; nbt++) {
    //  std::cout << "block=" << bx1block[nbt] << " rank=" << bx1ranks[nbt] << " lx2=" << loc.lx2 << " lx3=" << loc.lx3 << std::endl;
    //}  
    // At this point, each rank knows which blocks to sum over.
    // Now we need to clean the rank list (removing duplicates).
    int nrx1count = 0;
    int* rcount = new int[Globals::nranks];
    for (int nr=0; nr<Globals::nranks; nr++) {
      rcount[nr] = 0;
      for (int nbt=0; nbt<nbx1count; nbt++) { 
        if (bx1ranks[nbt] == nr) {
          rcount[nr] = 1;
        }
      }
      nrx1count += rcount[nr];
    }
    int irx1count = 0;
    int* rx1ranks = new int[nrx1count];
    for (int nr=0; nr<Globals::nranks; nr++) {
      if (rcount[nr] == 1) { // if rank is in global list for this column...
        rx1ranks[irx1count] = nr;  // ... store in communicator list
        irx1count++; // cannot have more than one rank per block.
      }
    }
    for (int irx1=0; irx1<nrx1count; irx1++) {
      std::cout << "cleaned rank list: rank=" << Globals::my_rank << " irx1=" << irx1 << " rx1ranks=" << rx1ranks[irx1] << std::endl;
    }
#ifdef MPI_PARALLEL
    int mpierr;
    mpierr = MPI_Comm_group(MPI_COMM_WORLD, &comm_x1.gworld);
    mpierr = MPI_Group_incl(comm_x1.gworld,nrx1count,rx1ranks,&comm_x1.gsub);
    mpierr = MPI_Comm_create(MPI_COMM_WORLD,comm_x1.gsub,&comm_x1.comsub);
    // now we have the communicator to sum along the x1 coordinate.
#endif /* MPI_PARALLEL */
    delete [] rx1ranks;
    delete [] rcount;
    delete [] bx1block;
    delete [] bx1ranks;
  }
#endif // WURSTBROT

}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief otf diagnostics (shell position, sphericity etc)
//========================================================================================

void MeshBlock::UserWorkInLoop(void) {
  return;
}

//========================================================================================
//! \fn void MeshBlock::OTFWorkBeforeOutput(void)
//  \brief otf diagnostics (shell position, sphericity etc)
//========================================================================================

void MeshBlock::OTFWorkBeforeOutput(ParameterInput *pin) {

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Any clean-up etc.
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  MeshBlock *pmb = pblock;
  //delete [] pmb->otf_data.data;
  return;
}
