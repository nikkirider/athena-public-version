//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file exp_default_pgen.cpp
//  \brief Provides default (empty) versions of all functions in problem generator files
//  This means user does not have to implement these functions if they are not needed.
//
//

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"

#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if (NSCALARS != 1)
#error: Requires NSCALARS = 1
#endif

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);
//Global variables for GridUpdate
int iexpupdate, iqunt;
int maxntrack = 20;
int ncycold=-1;
Real vtrack0;
AthenaArray<Real> ttrack,rtrack;

//Global Variables for OuterX1
Real ambDens;
Real ambVel;
Real ambPres;
Real b0, bz0, angle;
void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX2_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX3_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX2_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX3_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void ShockDetector(AthenaArray<Real> data, AthenaArray<Real> grid, int outArr[], Real eps);

//========================================================================================
//! \fn void WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData)
//  \brief Function that returns the velocity of cell wall i at location xf. Time, total
//  time step and direction are all given. Direction is one of 0,1,2, corresponding to x1,x2,x3
//  and gridData is an athena array that contains overall mesh data. gridData is updated
//  before every time sub-step by the UpdateGridData function. Some instances do not need
//  this data to be updated and the UpdateGridData function can be left blank. The gridData
//  array is supposed to carry all mesh-level information, i.e. the information used for 
//  multiple cell walls in the simulation.
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData) {
  Real retVal = 0.0;
  
 
  Real myX = xf;
  if (COORDINATE_SYSTEM == "cartesian") {
    if (dir == gridData(1)){
      if ((myX > 0.0)&&(gridData(3)>0.0)){ 
        if (gridData(2)==0.0) retVal = 0.0;
        else retVal = gridData(2) * myX/gridData(3);
      } else if ((myX < 0.0)&&(gridData(0)<0.0)){ 
        if (gridData(2) == 0.0) retVal = 0.0;
        else retVal = -1.0*gridData(2) * myX/gridData(0);
      }
    } else if (dir == gridData(5)) {
      if ((myX > 0.0)&&(gridData(7)>0.0)){ 
        if (gridData(6)==0.0) retVal = 0.0;
        else retVal = gridData(6) * myX/gridData(7);
      } else if ((myX < 0.0)&&(gridData(4)<0.0)){ 
        if (gridData(6) == 0.0) retVal = 0.0;
        else retVal = -1.0*gridData(6) * myX/gridData(4);
      }
    } else if (dir == gridData(9)) {
      if ((myX > 0.0)&&(gridData(11)>0.0)){ 
        if (gridData(10)==0.0) retVal = 0.0;
        else retVal = gridData(10) * myX/gridData(11);
      } else if ((myX < 0.0)&&(gridData(8)<0.0)){ 
        if (gridData(10) == 0.0) retVal = 0.0;
        else retVal = -1.0*gridData(10) * myX/gridData(8);
      }
    }   
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    if (dir != gridData(1)){
      retVal = 0.0;
    } else if (myX<=gridData(0)){
      retVal = 0.0;
    } else if (myX > gridData(0)){ 
      if (gridData(2)==0.0) retVal = 0.0;
      else retVal = gridData(2) * (myX-gridData(0))/(gridData(3)-gridData(0));
    } 
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    if (dir != gridData(1)){
      retVal = 0.0;
    } else if (myX<=gridData(0)){
      retVal = 0.0;
    } else if (myX > gridData(0)){ 
      if (gridData(2)==0.0) retVal = 0.0;
      else retVal = gridData(2) * (myX-gridData(0))/(gridData(3)-gridData(0));
    } 
  }

  return retVal; 
}

//========================================================================================
//! \fn void UpdateGridData(Mesh *pm)
//  \brief Function which can edit and calculate any terms in gridData, which is used 
//  in the WallVel function. The object in mesh is GridData(i) and i can range over the
//  integers, limited by SetGridData argument in InitMeshUserData. See exp_blast for an 
//  example use of this function.
//  This is an example function, showing various options of expansion tracking.
//  iepxupdate == 0: requires scalar field for mass-averaged radial velocity. 
//  iexpupdate == 1: gradient-weighted radial velocity. 
//    iqunt      == 0: pressure gradient 
//    iqunt      == 1: radial velocity gradient
//  Since the expansion velocity is calculated at the location of the shell, it needs
//  to be rescaled by xmax/mrad, where mrad is the (weighted) radius corresponding to the 
//  location of vrad. 
//========================================================================================
void UpdateGridData(Mesh *pm) {
  MeshBlock *pmb = pm->pblock;
  Real vtrack = 0.0, mrad = 0.0;
  Real gamma  = pmb->peos->GetGamma();
  int ntr=0;

  if (iexpupdate == -1) { // constant velocity 
    vtrack = vtrack0;
    if (Globals::my_rank == 0)
      fprintf(stdout,"[UpdateGrid]: vtrack=%13.5e xmax =%13.5e\n", vtrack,pm->mesh_size.x1max);
  } else if (iexpupdate == 0) { // needs scalar field in u(NHYDRO-1,...), calculates mass-weighted 
                         // velocity of mass shell. Works best for dense shells (snow-plow).
    Real mass = 0.0;
    if (COORDINATE_SYSTEM == "cartesian") {
      pm->GridData(3)  = pm->mesh_size.x1max;
      pm->GridData(0)  = pm->mesh_size.x1min;
      pm->GridData(7)  = pm->mesh_size.x2max;
      pm->GridData(4)  = pm->mesh_size.x2min;
      pm->GridData(11) = pm->mesh_size.x3max;
      pm->GridData(8)  = pm->mesh_size.x3min;
      Real rad = 0.0;
      Real x,y,z;
      while (pmb != NULL) {
        for (int k=pmb->ks; k<=pmb->ke; ++k) {
          z = pmb->pcoord->x3v(k);
          for (int j=pmb->js; j<=pmb->je; ++j) {
            y = pmb->pcoord->x2v(j);
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              Real v  = pmb->pcoord->GetCellVolume(k,j,i);
              Real c  = pmb->phydro->u(NHYDRO-1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
              x       = pmb->pcoord->x1v(i); 
              rad     = std::sqrt(SQR(x)+SQR(y)+SQR(z));
              vtrack +=  (  pmb->phydro->u(IM1,k,j,i)*x 
                          + pmb->phydro->u(IM2,k,j,i)*y 
                          + pmb->phydro->u(IM3,k,j,i)*z)
                        *c*v/rad;
              mass   += pmb->phydro->u(NHYDRO-1,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i); 
              mrad   += rad * pmb->phydro->u(NHYDRO-1,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
            }
          }
        }
        pmb = pmb->next;
      }
    } else { // if (COORDINATE_SYSTEM == "cartesian") 
      pm->GridData(3) = pm->mesh_size.x1max;
      while (pmb != NULL) {
        for (int k=pmb->ks; k<=pmb->ke; ++k) {
          for (int j=pmb->js; j<=pmb->je; ++j) {
            for (int i=pmb->is; i<=pmb->ie; ++i) {
              Real v  = pmb->pcoord->GetCellVolume(k,j,i);
              Real c  = pmb->phydro->u(NHYDRO-1,k,j,i)/pmb->phydro->u(IDN,k,j,i);
              vtrack += pmb->phydro->u(IM1,k,j,i)*c*v;
              mass   += pmb->phydro->u(NHYDRO-1,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
              mrad   += pmb->pcoord->x1v(i) * pmb->phydro->u(NHYDRO-1,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
            }
          }
        }
        pmb = pmb->next;
      }
    }
#ifdef MPI_PARALLEL
    Real myval[3];
    myval[0] = vtrack;
    myval[1] = mass;
    myval[2] = mrad;
    MPI_Allreduce(MPI_IN_PLACE,&myval,3,MPI_ATHENA_REAL,MPI_SUM,
                  MPI_COMM_WORLD);
    vtrack = myval[0];
    mass   = myval[1];
    mrad   = myval[2];
#endif
    mrad   /= mass;
    vtrack /= mass;
    vtrack  = ((vtrack < 0.0) ? 0.0 : vtrack);
    vtrack *= (pm->GridData(3) / mrad); // "boost" velocity to grid size
    if (Globals::my_rank == 0)
      fprintf(stdout,"[UpdateGrid] vtrack=%13.5e mass=%13.5e mrad=%13.5e xmax =%13.5e\n", vtrack,mass,mrad,pm->GridData(3));
  } else { // shock detector. Project gradient on radius
    Real totgrdr = 0.0;
    if (COORDINATE_SYSTEM=="cartesian") { 
      pm->GridData(3)  = pm->mesh_size.x1max;
      pm->GridData(0)  = pm->mesh_size.x1min;
      pm->GridData(7)  = pm->mesh_size.x2max;
      pm->GridData(4)  = pm->mesh_size.x2min;
      pm->GridData(11) = pm->mesh_size.x3max;
      pm->GridData(8)  = pm->mesh_size.x3min;
      AthenaArray<Real> qunt, grdr; 
      while (pmb != NULL) {
        int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
        if (pmb->block_size.nx3 > 1) {
          qunt.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
          grdr.NewAthenaArray(pmb->block_size.nx3+2*NGHOST,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        } else {
          qunt.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
          grdr.NewAthenaArray(1,pmb->block_size.nx2+2*NGHOST,pmb->block_size.nx1+2*NGHOST);
        }
        if (iqunt == 0) {
          if (DUAL_ENERGY) { // pressure as indicator
            for (int k=ks; k<=ke; ++k) {
              for (int j=js; j<=je; ++j) {
#pragma omp simd
                for (int i=is; i<=ie; ++i) {
                  qunt(k,j,i) = pmb->phydro->u(IIE,k,j,i)*(gamma-1.0);
                }
              }
            }
          } else {
            for (int k=ks; k<=ke; ++k) {
              for (int j=js; j<=je; ++j) {
#pragma omp simd
                for (int i=is; i<=ie; ++i) {
                  qunt(k,j,i) = (  pmb->phydro->u(IEN,k,j,i)
                                 - 0.5*(  SQR(pmb->phydro->u(IM1,k,j,i))
                                        + SQR(pmb->phydro->u(IM2,k,j,i))
                                        + SQR(pmb->phydro->u(IM3,k,j,i)))
                                      / pmb->phydro->u(IDN,k,j,i))*(gamma-1.0);
                }
                if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
                  for (int i=is; i<=ie; ++i) {
                    qunt(k,j,i) -= 0.5*(  SQR(pmb->pfield->bcc(IB1,k,j,i))
                                        + SQR(pmb->pfield->bcc(IB2,k,j,i))
                                        + SQR(pmb->pfield->bcc(IB3,k,j,i)))
                                      * (gamma-1.0);
                  }
                }
              }
            }
          }
        } else if (iqunt == 1) { // vrad as indicator
          for (int k=ks; k<=ke; ++k) {
            Real z = pmb->pcoord->x3v(k);
            for (int j=js; j<=je; ++j) {
              Real y = pmb->pcoord->x2v(j);
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                Real x   = pmb->pcoord->x1v(i);
                Real rad = std::sqrt(SQR(x)+SQR(y)+SQR(z));
                qunt(k,j,i) =  (  pmb->phydro->u(IM1,k,j,i)*x
                                + pmb->phydro->u(IM2,k,j,i)*y
                                + pmb->phydro->u(IM3,k,j,i)*z)
                             /(rad*pmb->phydro->u(IDN,k,j,i));
                qunt(k,j,i) = ((qunt(k,j,i) < 0.0) ? 0.0 : qunt(k,j,i));
              }
            }
          }
        }
        if (pmb->block_size.nx3 > 1) {
          for (int k=ks+1; k<=ke-1; ++k) {
            Real z = pmb->pcoord->x3v(k);
            Real zm= pmb->pcoord->x3v(k-1);
            Real zp= pmb->pcoord->x3v(k+1);
            for (int j=js+1; j<=je-1; ++j) {
              Real y = pmb->pcoord->x2v(j);
              Real ym= pmb->pcoord->x2v(j-1);
              Real yp= pmb->pcoord->x2v(j+1);
              for (int i=is+1; i<=ie-1; ++i) {
                Real x      = pmb->pcoord->x1v(i);
                Real xm     = pmb->pcoord->x1v(i-1);
                Real xp     = pmb->pcoord->x1v(i+1);
                Real rad    = std::sqrt(SQR(x)+SQR(y)+SQR(z));
                Real gx     = (qunt(k  ,j  ,i+1)-qunt(k  ,j  ,i-1))/(xp-xm);
                Real gy     = (qunt(k  ,j+1,i  )-qunt(k  ,j-1,i  ))/(yp-ym);
                Real gz     = (qunt(k+1,j  ,i  )-qunt(k-1,j  ,i  ))/(zp-zm);
                grdr(k,j,i) = (gx*x+gy*y+gz*z)/rad;
                grdr(k,j,i) = std::fabs(grdr(k,j,i)); //((grdr(k,j,i) < 0.0) ? -grdr(k,j,i) : 0.0);
                totgrdr    += grdr(k,j,i);        
                vtrack     +=  (  pmb->phydro->u(IM1,k,j,i)*x
                                + pmb->phydro->u(IM2,k,j,i)*y
                                + pmb->phydro->u(IM3,k,j,i)*z)
                              *grdr(k,j,i)/rad;
                mrad       += rad * grdr(k,j,i);
              }
            }
          }
        } else { // two dimensions
          for (int j=js+1; j<=je-1; ++j) {
            Real y = pmb->pcoord->x2v(j);
            Real ym= pmb->pcoord->x2v(j-1);
            Real yp= pmb->pcoord->x2v(j+1);
            for (int i=is+1; i<=ie-1; ++i) {
              Real x        = pmb->pcoord->x1v(i);
              Real xm       = pmb->pcoord->x1v(i-1);
              Real xp       = pmb->pcoord->x1v(i+1);
              Real rad      = std::sqrt(SQR(x)+SQR(y));
              Real gx       = (qunt(ks,j  ,i+1)-qunt(ks,j  ,i-1))/(xp-xm);
              Real gy       = (qunt(ks,j+1,i  )-qunt(ks,j-1,i  ))/(yp-ym);
              grdr(ks,j,i)  = (gx*x+gy*y)/rad; // emphasize outer gradient
              grdr(ks,j,i)  = std::fabs(grdr(ks,j,i)); //((grdr(ks,j,i) < 0.0) ? -grdr(ks,j,i) : 0.0);
              //grdr(ks,j,i) *= SQR(SQR(rad)*SQR(rad));
              totgrdr      += grdr(ks,j,i);
              vtrack       +=  (  pmb->phydro->u(IM1,ks,j,i)*x
                                + pmb->phydro->u(IM2,ks,j,i)*y)
                              *grdr(ks,j,i)/(pmb->phydro->u(IDN,ks,j,i)*rad);
              mrad         += rad * grdr(ks,j,i);
            }
          }
        }
        qunt.DeleteAthenaArray();
        grdr.DeleteAthenaArray();
        pmb = pmb->next;
      } // while (pmb != NULL)
    } else { // if (COORDINATE_SYSTEM=="cartesian")

    }
#ifdef MPI_PARALLEL
    Real myval[3];
    myval[0] = vtrack;
    myval[1] = totgrdr;
    myval[2] = mrad;
    MPI_Allreduce(MPI_IN_PLACE,&myval,3,MPI_ATHENA_REAL,MPI_SUM,
                  MPI_COMM_WORLD);
    vtrack  = myval[0];
    totgrdr = myval[1];
    mrad    = myval[2];
#endif
    mrad   /= totgrdr;

    if (iexpupdate == 1) { // use gradient-weighted radial velocity
      vtrack /= totgrdr;
      vtrack  = ((vtrack < 0.0) ? 0.0 : vtrack);
      vtrack *= (pm->GridData(3)/mrad); // boost velocity
    } else if (iexpupdate == 2) { // use position of gradient mrad
      
      ntr = (pm->ncycle >= maxntrack) ? maxntrack-1 : pm->ncycle;
      //fprintf(stdout,"[UpdateGridData]: ntr=%2i\n",ntr);
      
      if (ncycold < pm->ncycle) {
        ncycold = pm->ncycle;
        if (ntr == 0) {
          ttrack(0) = 0.0;
          rtrack(0) = mrad;
          vtrack    = 0.0;
        } else {
          for (int m=0; m<ntr; ++m) {  
            ttrack(ntr-m) = ttrack(ntr-m-1);
            rtrack(ntr-m) = rtrack(ntr-m-1);
            //fprintf(stdout,"   m=%2i ntr-m=%2i ttrack[ntr-m]=%13.5e rtrack[ntr-m]=%13.5e | ntr-m-1=%2i ttrack[ntr-m-1]=%13.5e rtrack[ntr-m-1]=%13.5e\n",
            //        m,ntr-m,ttrack(ntr-m),rtrack(ntr-m),ntr-m-1,ttrack(ntr-m-1),rtrack(ntr-m-1));
          }
          ttrack(0) = pm->time;
          rtrack(0) = mrad;
          //for (int m=0; m<ntr; ++m) {
          //  fprintf(stdout,"   m=%2i ttrack=%10.2e rtrack=%10.2e\n",m,ttrack(m),rtrack(m));
          //}
        }
      }
     //  else {
     //   fprintf(stdout,"[UpdateGridData]: Use old info.\n");
     // }

      vtrack = 0.0;

      for (int m=1; m<ntr; ++m) {
        Real vt = (rtrack(m-1)-rtrack(m))/(ttrack(m-1)-ttrack(m)) * (pm->GridData(3)/mrad);
        vtrack += vt;
        //fprintf(stdout,"   m=%2i r[m-1]-r[m]=%13.5e t[m-1]-t[m]=%13.5e vtrack[m]=%13.5e\n",m,rtrack(m-1)-rtrack(m),ttrack(m-1)-ttrack(m),vt);
      }
      if (ntr > 0) vtrack /= ntr;
      //fprintf(stdout,"[UpdateGridData]:    ntr=%3i t=%13.5e vtrack=%13.5e mrad=%13.5e ttrack=%13.5e %13.5e %13.5e %13.5e rtrack=%13.5e %13.5e %13.5e %13.5e\n",
      //        ntr,pm->time,vtrack,mrad,ttrack(0),ttrack(1),ttrack(2),ttrack(3),rtrack(0),rtrack(1),rtrack(2),rtrack(3));
    }
    vtrack = (vtrack <= 0.0) ? 0.0 : vtrack;
    if (Globals::my_rank == 0)
      fprintf(stdout,"[UpdateGrid] vtrack=%13.5e totgrdr=%13.5e mrad=%13.5e xmax =%13.5e\n", vtrack,totgrdr,mrad,pm->GridData(3));
  }

  pm->GridData(2) = vtrack;
  if (COORDINATE_SYSTEM=="cartesian") {
    pm->GridData(6) = vtrack;
    pm->GridData(10) = vtrack;
  }

  return;
}


//Harten Van Leer Shock detection algorithm Out data should be a 1 dimensional array,
// with the same length as indata and grid. indata is the array of Real values where
// we look for the shocks. eps is the slope magnitude limiter, i.e. if the slope is
// above eps, then the location has a shock.
void ShockDetector(AthenaArray<Real> data, AthenaArray<Real> grid, int outArr[], Real eps ) {
  int n, loc;
  Real a, b, c;
  n = data.GetDim1();
  AthenaArray<Real> shockData;
  shockData.NewAthenaArray(n-1);
  loc = 0;
  for (int i=1; i<(n-1); ++i) {
    a = 0;
    b = 0;
    a = std::abs(data(i)-data(i-1));
    b = std::abs(data(i+1)-data(i));
    c = a+b;
    shockData(i-1) = SQR(a-b);

    if ( c <= eps) { 
      shockData(i-1) = 0.0;
    } else { 
      shockData(i-1) /= SQR(a+b);
    }  
  }  
  int k=0;
  for (int i=0; i< (n-1); ++i) {
    if (shockData(i) >= 0.95){
      outArr[k] = i;
      k+=1;    
    }

  }
 
  shockData.DeleteAthenaArray();
  return;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  //========================================================================================
  //! \brief For a time dependent grid, make sure to use SetGridData, EnrollGridDiffEq, and
  //   EnrollCalcGridData here. The boundary conditions are of course optional. Reflecting 
  //   is a good boundary function if a wall of the simulation is static. But if there is
  //   any expansion of the grid, it is recommended that you use the UniformMedium condition
  //   for the expanding boundary. Otherwise, reconstruction might fail because the data is
  //   inaccurate (for example, periodic boundary conditions do not make sense
  //   for an expanding grid).
  //========================================================================================
  if (EXPANDING_ENABLED) {
    EnrollGridDiffEq(WallVel);
      
    if (COORDINATE_SYSTEM == "cartesian") {
      SetGridData(12);

      if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X1,OuterX1_UniformMedium);
      }
      if (mesh_bcs[OUTER_X2] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X2,OuterX2_UniformMedium);
      }
      if (mesh_bcs[OUTER_X3] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X3,OuterX3_UniformMedium);
      }

      if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X1,InnerX1_UniformMedium);
      }
      if (mesh_bcs[INNER_X2] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X2,InnerX2_UniformMedium);
      }
      if (mesh_bcs[INNER_X3] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(INNER_X3,InnerX3_UniformMedium);
      }

    } else {
      SetGridData(4);
      if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
        EnrollUserBoundaryFunction(OUTER_X1,OuterX1_UniformMedium);
      }
    }
    EnrollCalcGridData(UpdateGridData);
    ttrack.NewAthenaArray(maxntrack); // for position tracking
    rtrack.NewAthenaArray(maxntrack);
    
    ambDens    = pin->GetReal("problem","damb");
    ambVel     = 0.0;
    ambPres    = pin->GetReal("problem","pamb");
    iexpupdate = pin->GetOrAddInteger("problem","iexpupdate",0); // 0: mass-weighted velocity, 1: shock position
    iqunt      = pin->GetOrAddInteger("problem","iqunt",1); // 0: pressure as indicator, 1: vrad as indicator

    if (iexpupdate == -1) {
      vtrack0 = pin->GetReal("problem","vtrack0"); // constant tracking velocity for test purposes
    }
    
    Real rout  = pin->GetReal("problem","radius");
    Real rin   = rout - pin->GetOrAddReal("problem","ramp",0.0);
    Real vs    = pin->GetOrAddReal("problem","vel",0.0);

    if (COORDINATE_SYSTEM == "cartesian") {
      GridData(0) = mesh_size.x1min;
      GridData(1) = 1; 
      GridData(2) = 0.0;
      GridData(3) = mesh_size.x1max; 

      GridData(4) = mesh_size.x2min;
      GridData(5) = 2; 
      GridData(6) = 0.0;
      GridData(7) = mesh_size.x2max; 

      GridData(8) = mesh_size.x3min;
      GridData(9) = 3; 
      GridData(10) = 0.0;
      GridData(11) = mesh_size.x3max; 
    } else {
      GridData(0) = mesh_size.x1min;
      GridData(1) = 1; 
      GridData(2) = 0.0;
      GridData(3) = mesh_size.x1max; 
    }
  }

  return;
}

//========================================================================================
//! \fn void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i) = ambDens;
        prim(IPR,k,j,ie+i) = ambPres;  
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,ie+i+1) = b0 * std::cos(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,ie+i+1) = b0 * (   std::cos(angle) * std::cos(phi)
                                       + std::sin(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,ie+i+1) = b0 * std::abs(std::sin(theta))
                                   * (   std::cos(angle) * std::cos(phi)
                                       + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=js; j<=je+1; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,ie+i) = b0 * std::sin(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,ie+i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") 
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,ie+i) = b0 * std::cos(theta)
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              b.x2f(k,j,ie+i) *= -1.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,ie+i) = bz0;
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,ie+i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
  }

  return;

}


//========================================================================================
//! \fn void OuterX2_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX2_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int i=is; i<=ie; ++i) {
#pragma omp simd
      for (int j=1; j<=ngh; ++j) {
        prim(IDN,k,je+j,i) = ambDens;
        prim(IPR,k,je+j,i) = ambPres;  
        prim(IVX,k,je+j,i) = 0.0;
        prim(IVY,k,je+j,i) = 0.0;
        prim(IVZ,k,je+j,i) = 0.0;
      }
    }
  }


  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=1; j<=ngh; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(k,je+j,i) = b0 * std::cos(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(je+j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(k,je+j,i) = b0 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
          theta = pco->x2v(je+j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(k,je+j,i) = b0 * std::abs(std::sin(theta))
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=1; j<=ngh; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(k,je+j+1,i) = b0 * std::sin(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(je+j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(k,je+j+1,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                       - std::cos(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") 
          theta = pco->x2v(je+j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(k,je+j+1,i) = b0 * std::cos(theta)
                                   * (   std::cos(angle) * std::cos(phi)
                                       + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              b.x2f(k,je+j+1,i) *= -1.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=1; j<=ngh; ++j) {
        if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(k,je+j,i) = bz0;
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(k,je+j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
  }

  return;

}

//========================================================================================
//! \fn void OuterX3_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for outer boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================
void OuterX3_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int j=js; j<=je; ++j) {
    for (int i=is; i<=ie; ++i) {
#pragma omp simd
      for (int k=1; k<=ngh; ++k) {
        prim(IDN,ke+k,j,i) = ambDens;
        prim(IPR,ke+k,j,i) = ambPres;  
        prim(IVX,ke+k,j,i) = 0.0;
        prim(IVY,ke+k,j,i) = 0.0;
        prim(IVZ,ke+k,j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=1; k<=ngh; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(ke+k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ke+k,j,i) = b0 * std::cos(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ke+k,j,i) = b0 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ke+k,j,i) = b0 * std::abs(std::sin(theta))
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(ke+k);
      for (int j=js; j<=je+1; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(ke+k,j,i) = b0 * std::sin(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(ke+k,j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") 
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(ke+k,j,i) = b0 * std::cos(theta)
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              b.x2f(ke+k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(ke+k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(ke+k+1,j,i) = bz0;
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(ke+k+1,j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                       - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
  }

  return;

}
//========================================================================================
//! \fn void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = ambDens;
        prim(IPR,k,j,is-i) = ambPres;  
        prim(IVX,k,j,is-i) = 0.0;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar") 
        phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,is-i) = b0 * std::cos(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,is-i) = b0 * (   std::cos(angle) * std::cos(phi) 
                                     + std::sin(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x1f(k,j,is-i) = b0 * std::abs(std::sin(theta))
                                 * (   std::cos(angle) * std::cos(phi) 
                                     + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=js; j<=je+1; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,is-i) = b0 * std::sin(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,is-i) = b0 * (   std::sin(angle) * std::cos(phi) 
                                     - std::cos(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") 
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x2f(k,j,is-i) = b0 * std::cos(theta)
                                 * (   std::cos(angle) * std::cos(phi) 
                                     + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              b.x2f(k,j,is-i) *= -1.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,is-i) = bz0;
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            b.x3f(k,j,is-i) = b0 * (   std::sin(angle) * std::cos(phi) 
                                     - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
  }
  return;

}
//========================================================================================
//! \fn void InnerX2_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX2_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,k,js-j,i) = ambDens;
        prim(IPR,k,js-j,i) = ambPres;  
        prim(IVX,k,js-j,i) = 0.0;
        prim(IVY,k,js-j,i) = 0.0;
        prim(IVZ,k,js-j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=1; j<=ngh; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(k,js-j,i) = b0 * std::cos(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(js-j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(k,js-j,i) = b0 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
          theta = pco->x2v(js-j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(k,js-j,i) = b0 * std::abs(std::sin(theta))
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=1; j<=ngh; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(k,js-j,i) = b0 * std::sin(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(js-j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(k,js-j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") 
          theta = pco->x2v(js-j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(k,js-j,i) = b0 * std::cos(theta)
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              b.x2f(k,js-j,i) *= -1.0;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(k);
      for (int j=1; j<=ngh; ++j) {
        if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(k,js-j,i) = bz0;
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(k,js-j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
  }

  return;

}

//========================================================================================
//! \fn void InnerX3_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX3_UniformMedium(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ks-k,j,i) = ambDens;
        prim(IPR,ks-k,j,i) = ambPres;  
        prim(IVX,ks-k,j,i) = 0.0;
        prim(IVY,ks-k,j,i) = 0.0;
        prim(IVZ,ks-k,j,i) = 0.0;
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    Real theta, phi;
    for (int k=1; k<=ngh; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(ks-k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ks-k,j,i) = b0 * std::cos(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ks-k,j,i) = b0 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            b.x1f(ks-k,j,i) = b0 * std::abs(std::sin(theta))
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(ks-k);
      for (int j=js; j<=je+1; ++j) {
        if (COORDINATE_SYSTEM == "cartesian") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(ks-k,j,i) = b0 * std::sin(angle);
          }
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phi = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(ks-k,j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") 
          theta = pco->x2v(j);
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x2f(ks-k,j,i) = b0 * std::cos(theta)
                                 * (   std::cos(angle) * std::cos(phi)
                                     + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              b.x2f(ks-k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      if (COORDINATE_SYSTEM == "spherical_polar")
        phi = pco->x3v(ks-k);
      for (int j=js; j<=je; ++j) {
        if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(ks-k,j,i) = bz0;
          }
        } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            b.x3f(ks-k,j,i) = b0 * (   std::sin(angle) * std::cos(phi)
                                     - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
  }

  return;

}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // In practice, this function should *always* be replaced by a version
  // that sets the initial conditions for the problem of interest.
  Real rout = pin->GetReal("problem","radius");
  Real dr  =  pin->GetReal("problem","ramp");
  Real pa   = pin->GetReal("problem","pamb");
  Real da   = pin->GetReal("problem","damb");
  Real prat = pin->GetReal("problem","prat");
  Real drat = pin->GetReal("problem","drat");
  Real vSh   = pin->GetReal("problem","vel");
  if (MAGNETIC_FIELDS_ENABLED) {
    b0 = pin->GetReal("problem","b0");
    bz0= pin->GetOrAddReal("problem","bz0",0.0);
    angle = (PI/180.0)*pin->GetReal("problem","angle");
    if (COORDINATE_SYSTEM == "spherical_polar")
      bz0 = 0.0;
  }
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;

  //fprintf(stdout,"IDN=%2i IVX=%2i IVY=%2i IVZ=%2i IPR=%2i IBY=%2i IBZ=%2i NHYDRO-SCALARS=%2i NHYDRO=%2i NWAVE=%2i\n",IDN,IVX,IVY,IVZ,IPR,IBY,IBZ,NHYDRO-NSCALARS,NHYDRO,NWAVE);

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x1_0   = pin->GetOrAddReal("problem","x1_0",0.0);
  Real x2_0   = pin->GetOrAddReal("problem","x2_0",0.0);
  Real x3_0   = pin->GetOrAddReal("problem","x3_0",0.0);
  Real x0,y0,z0;
  if (COORDINATE_SYSTEM == "cartesian") {
    x0 = x1_0;
    y0 = x2_0;
    z0 = x3_0;
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    x0 = x1_0*std::cos(x2_0);
    y0 = x1_0*std::sin(x2_0);
    z0 = x3_0;
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::cout << "### FATAL ERROR in blast.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
  }

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad, r, x, y, z;
        if (COORDINATE_SYSTEM == "cartesian") {
          x   = pcoord->x1v(i);
          y   = pcoord->x2v(j);
          z   = pcoord->x3v(k);
          r   = std::sqrt(SQR(x)+SQR(y)+SQR(z));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          x   = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          y   = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          z   = pcoord->x3v(k);
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        } else { // if (COORDINATE_SYSTEM == "spherical_polar")
          x   = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          y   = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          z   = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        }
        Real den = da;
        Real v1  = 0.0;
 
        den += da*(drat-1.0)*0.5*(1.0-std::tanh((rad-rout)/dr));
        v1  += vSh*0.5*(1.0-std::tanh((rad-rout)/dr))*(rad/rout);
 
        phydro->u(IDN,k,j,i) = den;
        if (COORDINATE_SYSTEM == "cartesian") {
          phydro->u(IM1,k,j,i) = den*v1*x/r;
          phydro->u(IM2,k,j,i) = den*v1*y/r;
          phydro->u(IM3,k,j,i) = den*v1*z/r;
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          phydro->u(IM1,k,j,i) = den*v1;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        } else if (COORDINATE_SYSTEM == "spherical_polar") {
          phydro->u(IM1,k,j,i) = den*v1;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
        if (NON_BAROTROPIC_EOS) {
          Real pres = pa;
          pres += pa*(prat-1.0)*0.5*(1.0-std::tanh((rad-rout)/dr));
          phydro->u(IEN,k,j,i) = 0.5*den*SQR(v1)+pres/gm1;
        }
 
        for (int n=NHYDRO-NSCALARS; n<NHYDRO; ++n) {
          phydro->u(n,k,j,i) = den*0.25*(1.0+std::tanh((rad-1.0*rout)/(0.01*rout)))
                                       *(1.0-std::tanh((rad-1.2*rout)/(0.01*rout)));
        }
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x1f(k,j,i) = b0 * std::cos(angle);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x1f(k,j,i) =
                b0 * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x1f(k,j,i) = b0 * std::abs(std::sin(theta))
                * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x2f(k,j,i) = b0 * std::sin(angle);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x2f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x2f(k,j,i) = b0 * std::cos(theta)
                * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              pfield->b.x2f(k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
            pfield->b.x3f(k,j,i) = bz0;
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real phi = pcoord->x3v(k);
            pfield->b.x3f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*(SQR(b0)+SQR(bz0));
        }
      }
    }
  } 

  return;
}

//====================================================================================

void Mesh::UserWorkInLoop(void) {
  return;
}
          
