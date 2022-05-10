//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file magshell.cpp
//  \brief Expanding magnetic shell.
//
//

#include <algorithm>
#include <cmath>
#include <cfloat>     // FLT_MAX
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
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

//========================================================================================
// Radiative loss functions
//========================================================================================

typedef Real (*CoolingFunc_t)(const Real dens, const Real temp); // For generic cooling functions

int icool, iturb;
Real gm1;
Real coolsafe = 0.05;
Real CoolingFuncSlyz(const Real dens, const Real temp);
Real CoolingFuncSlyzMod(const Real dens, const Real temp);
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt);
Real BracketRoot(const Real dens, const Real temp0, const Real dt);
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt);
Real GetEquiTemp(const Real dens,const Real temp0);
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
CoolingFunc_t CoolingFunc;
static void stop_this();

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);

//Global Variables for OuterX1
Real damb,pamb; // ambient parameters
Real x1cld=0.0,x2cld=0.0,x3cld=0.0,dcld=0.0,rcld=0.0; // cloud parameters
Real x1fil=0.0,x2fil=0.0,x3fil=0.0,rfil=0.0,lfil=0.0,dfil=0.0; // filament parameters
Real bx0,by0,bz0;
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
// short for debugging interrupt
//========================================================================================
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//========================================================================================
// Real CoolingFuncSlyz(const Real dens, const Real temp)
//   Cooling function from Slyz et al. 2005 (piece-wise powerlaw fit)
//   Returns de/dt [ergs/s] in code units assuming ISM units with n0=T0=1.
//========================================================================================
Real CoolingFuncSlyz(const Real dens, const Real temp) {
  const Real fac = 2.167177868e+31;
  Real lambda, dedt;
  if (temp < 3.1e2) {
    lambda = 0.0;
  } else if (temp < 2.0e3) {
    lambda = 2.2380e-32*SQR(temp);
  } else if (temp < 8.0e3) {
    lambda = 1.0012e-30*std::pow(temp,1.5);
  } else if (temp < 3.9811e4) {
    lambda = 4.6240e-36*std::pow(temp,2.867);
  } else if (temp < 1.0e5) {
    lambda = 3.162e-30*std::pow(temp,1.6);
  } else if (temp < 2.884e5) {
    lambda = 3.162e-21*std::pow(temp,-0.2);
  } else if (temp < 4.732e5) {
    lambda = 6.3100e-6*std::pow(temp,-3.0);
  } else if (temp < 2.113e6) {
    lambda = 1.047e-21*std::pow(temp,-0.22);
  } else if (temp < 3.981e6) {
    lambda = 3.981e-4*std::pow(temp,-3.0);
  } else if (temp < 1.995e7) {
    lambda = 4.169e-26*std::pow(temp,0.33);
  } else {
    lambda = 2.399e-27*std::sqrt(temp);
  }
  dedt = -dens*lambda*fac;
  return dedt;
}

//========================================================================================
// Real CoolingFuncSlyzMod(const Real dens, const Real temp)
//   Cooling function from Slyz et al. 2005 (piece-wise powerlaw fit)
//   Returns de/dt [ergs/s] in code units assuming ISM units with n0=T0=1.
//========================================================================================
Real CoolingFuncSlyzMod(const Real dens, const Real temp) {
  const Real fac = 2.167177868e+31;
  Real gamma, lambda, dedt, T0=2.0e1, T1=1.0e2, T2=8.0e4, G0=5.0e-25;
  // From yacc.py
  //  i= 0 a=  1.89059e-31 T=  3.00000e+00 b=  3.00000e+00 L=  7.87204e-27
  //  i= 1 a=  4.72648e-28 T=  5.00000e+01 b=  1.00000e+00 L=  2.36324e-26
  //  i= 2 a=  1.18724e-25 T=  1.00000e+03 b=  2.00000e-01 L=  4.72648e-25
  //  i= 3 a=  4.62400e-36 T=  8.00000e+03 b=  2.86700e+00 L=  7.16400e-25
  //  i= 4 a=  3.16200e-30 T=  3.98110e+04 b=  1.60000e+00 L=  7.13225e-23
  //  i= 5 a=  3.16200e-21 T=  1.00000e+05 b= -2.00000e-01 L=  3.16200e-22
  //  i= 6 a=  3.96238e-14 T=  2.88400e+05 b= -1.50000e+00 L=  2.55837e-22
  //  i= 7 a=  1.21075e-01 T=  4.73200e+05 b= -3.70000e+00 L=  1.21727e-22
  //  i= 8 a=  4.82008e-25 T=  1.00000e+06 b=  2.00000e-01 L=  7.63931e-24
  //  i= 9 a=  2.44135e-26 T=  3.00000e+06 b=  4.00000e-01 L=  9.51652e-24
  //  i=10 a=  4.54493e-27 T=  2.00000e+07 b=  5.00000e-01 L=  2.03255e-23
  if (temp < 3.0e0) { 
    lambda = 0.0;
  } else if (temp < 5.0e1) { // i=0
    lambda = 1.89059e-31*SQR(temp)*temp;
  } else if (temp < 1.0e3) { // i=1
    lambda = 4.72648e-28*temp;
  } else if (temp < 8.0e3) { // i=2
    lambda = 1.18724e-25*std::pow(temp,0.2);
  } else if (temp < 3.9811e4) { // i=3
    lambda = 4.6240e-36*std::pow(temp,2.867);
  } else if (temp < 1.0e5) { // i=4
    lambda = 3.162e-30*std::pow(temp,1.6);
  } else if (temp < 2.884e5) { // i=5
    lambda = 3.162e-21*std::pow(temp,-0.2);
  } else if (temp < 4.732e5) { // i=6
    lambda = 3.96238e-14*std::pow(temp,-1.5);
  } else if (temp < 1.0e6) { // i=7
    lambda = 1.21075e-01*std::pow(temp,-3.7);
  } else if (temp < 3.0e6) { // i=8
    lambda = 4.82008e-25*std::pow(temp,0.2);
  } else if (temp < 2.0e7) { // i=9
    lambda = 2.44135e-26*std::pow(temp,0.4);
  } else { // i=10
    lambda = 4.54493e-27*std::sqrt(temp);
  }
  if (temp < T0) {
    gamma = G0*(T1/T0)*std::exp(T0-temp);
  } else if (temp < T1) {
    gamma = G0*(T1/temp);
  } else if (temp < T2) {
    gamma = G0;
  } else {
    gamma = G0*(T2/temp);
  }
  dedt = (gamma-dens*lambda)*fac;
  return dedt;
}

//========================================================================================
// Real RootFunc(const Real dens, const Real temp0, const Real, temp1, const Real dt)
// This is not the thermal equilibrium, but the implicit update for the thermal ODE
//   dT = dt*(gamma-1)/kB * (Gamma(T)-n*Lambda(T)).
//   As long as dt is limited to a fraction of dtcool (see HeatCool), the implicit solution
//   prevents under- or over-shoots.
//========================================================================================
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt) {
  return temp0 + dt*gm1*CoolingFunc(dens,temp1) - temp1;
}

//========================================================================================
// Real BracketRoot(const Real temp0, const Real dt)
//========================================================================================
Real BracketRoot(const Real dens, const Real temp0, const Real dt) {
  Real rf    = RootFunc(dens,temp0,temp0,dt);
  Real sig   = (Real) ((rf > 0) - (rf < 0));
  Real fac   = 1.0 + sig*0.1;
  Real temp1 = temp0;
  while (rf*RootFunc(dens,temp0,temp1,dt) > 0)
    temp1 *= fac;
  return temp1;
}

//========================================================================================
// Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt)
// \brief Finds root for RootFunc via bisection. Version without if-statements.
//========================================================================================
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt) {
  if (CoolingFunc(dens,temp0) == 0.0) return temp0; // Nothing to do for thermal equilibrium
  // Otherwise, temp1 and temp0 bracket the temperature down to which we should integrate.
  const Real tol = 1e-6;
  int nit = (int) (log(fabs(temp1-temp0)/tol)/log(2.0));
  Real T[3], L[2];
  T[0]         = temp0;
  T[1]         = temp1;
  T[2]         = 0.5*(T[0]+T[1]);
  L[0]         = RootFunc(dens,temp0,T[0],dt);
  L[1]         = RootFunc(dens,temp0,T[2],dt);
  for (int i=0; i<nit; i++) {
    int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
    T[w]  = T[2];
    L[w]  = L[1];
    T[2]  = 0.5*(T[0]+T[1]);
    L[1]  = RootFunc(dens,temp0,T[2],dt);
  }
  return T[2];
}

//========================================================================================
// Real GetEquiTemp(const Real dens, const Real temp0)
// \brief Finds equilibrium temperature for cooling curve with equilibrium.
//   Contains bracketing step as well. Assumes that new temperature < temp0.
//========================================================================================
Real GetEquiTemp(const Real dens, const Real temp0) {
  Real T[3], L[2];
  Real fac, sig;
  const Real tol = 1e-6;
  int nit;
  L[0] = CoolingFunc(dens,temp0);
  if (L[0] == 0.0) return temp0;
  // bracket
  sig  = SIGN(L[0]); 
  fac  = 1.0 + sig*0.1;
  T[0] = temp0;
  T[1] = fac*T[0];
  L[1] = CoolingFunc(dens,T[1]);
  while (L[0]*L[1] > 0.0) {
    T[0] = T[1];
    T[1] = fac*T[0];
    L[0] = L[1];
    L[1] = CoolingFunc(dens,T[1]);
  }
  // root
  nit = (int) (log(fabs(T[1]-T[0])/tol)/log(2.0));
  T[2] = 0.5*(T[0]+T[1]);
  L[0] = CoolingFunc(dens,T[0]);
  L[1] = CoolingFunc(dens,T[2]);
  for (int i=0; i<nit; i++) {
    int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
    T[w]  = T[2];
    L[w]  = L[1];
    T[2]  = 0.5*(T[0]+T[1]);
    L[1]  = CoolingFunc(dens,T[2]);
  }
  return T[2];
}

//========================================================================================
//! \fn void heatcool(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//  Sets global variable dtcool to allow for adjustment of timestep if required 
//  (See HeatCoolTimeStep).
//  The current timestep at which change is applied is dt. This cannot be changed. Hence
//  any safety factor for dtcool needs to be aggressively applied for next timestep.
//  The only other remedy would be to implement a "traceback" as in Proteus and Athena. 
//========================================================================================
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{

  Real g1  = pmb->peos->GetGamma()-1.0;

  AthenaArray<Real> dens, temp0, temp1, temp2, dener, dtcool, edot, sign; 
  dens.NewAthenaArray(prim.GetDim1());
  temp0.NewAthenaArray(prim.GetDim1());
  temp1.NewAthenaArray(prim.GetDim1());
  temp2.NewAthenaArray(prim.GetDim1()); 
  dener.NewAthenaArray(prim.GetDim1()); // Delta E by which to change total energy
  dtcool.NewAthenaArray(prim.GetDim1());
  edot.NewAthenaArray(prim.GetDim1());
  sign.NewAthenaArray(prim.GetDim1());

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x32 = SQR(pmb->pcoord->x3v(k));
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x22 = SQR(pmb->pcoord->x2v(j));
      if (DUAL_ENERGY) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          dens(i)  = prim(IDN,k,j,i); 
          temp0(i) = prim(IGE,k,j,i)/dens(i); // IGE is pressure
        }
      } else {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          dens(i)  = prim(IDN,k,j,i); 
          temp0(i) = prim(IPR,k,j,i)/dens(i);
        } 
      }
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (temp0(i) <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in coolcloud.cpp: HeatCool: temp0 <=0" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0(i)
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens (i)<< std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      }
      if (icool == 2) { // CoolingFuncSlyzMod with equilibrium
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          edot(i)   = CoolingFunc(dens(i),temp0(i));
          temp1(i)  = GetEquiTemp(dens(i),temp0(i));
          sign(i)   = SIGN(temp0(i)-temp1(i));
          dtcool(i) = fabs(temp0(i)-temp1(i)) / (gm1*(fabs(edot(i))+1e-60));
          temp2(i)  = temp1(i) + (temp0(i)-temp1(i))*std::exp(-dt/dtcool(i));
        }
      } else { // CoolingFuncSlyz without equilibrium
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          temp1(i) = BracketRoot(dens(i),temp0(i),dt);
          temp2(i) = FindRoot(dens(i),temp0(i),temp1(i),dt);
        }
      }
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dener(i)        = dens(i)*(temp2(i)-temp0(i))/g1;
        cons(IEN,k,j,i)+= dener(i);
        if (DUAL_ENERGY)
          cons(IIE,k,j,i) += dener(i);
      }
    }
  }
  dens.DeleteAthenaArray();
  temp0.DeleteAthenaArray();
  temp1.DeleteAthenaArray();
  temp2.DeleteAthenaArray();
  dener.DeleteAthenaArray();
  dtcool.DeleteAthenaArray();
  edot.DeleteAthenaArray();
  sign.DeleteAthenaArray();

  return;
}

//========================================================================================
//! \fn void HeatCoolTimeStep(...)
//  \brief Calculates cooling timestep and sends it to new_blockdt
//    Parameter coolsafe works as "CFL" (safety) factor for cooling: Timestep 
//    reduced to ensure only coolsafe*100 % of internal energy is removed. 
//========================================================================================
Real HeatCoolTimeStep(MeshBlock *pmb)
{
  Real g1     = pmb->peos->GetGamma()-1.0;
  Real dtcool = HUGE_NUMBER;// allows it to grow

  AthenaArray<Real> dens, temp0, w;
  w.InitWithShallowCopy(pmb->phydro->w);
  dens.NewAthenaArray(w.GetDim1());
  temp0.NewAthenaArray(w.GetDim1());

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      if (DUAL_ENERGY) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          dens(i)  = w(IDN,k,j,i); 
          temp0(i) = w(IGE,k,j,i)/dens(i); // IGE is pressure
        } 
      } else {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          dens(i)  = w(IDN,k,j,i); 
          temp0(i) = w(IPR,k,j,i)/dens(i);
        } 
      } 
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dttemp   = coolsafe*temp0(i)/(fabs(CoolingFunc(dens(i),temp0(i)))+1e-60);
        if (TIMESTEPINFO_ENABLED) {
          if (dttemp < dtcool) {
            pmb->all_min_dts(8)   = dttemp;
            pmb->all_min_loc(8,0) = pmb->pcoord->x1v(i);
            pmb->all_min_loc(8,1) = pmb->pcoord->x2v(j);
            pmb->all_min_loc(8,2) = pmb->pcoord->x3v(k);
            pmb->all_min_ind(8,0) = i;
            pmb->all_min_ind(8,1) = j;
            pmb->all_min_ind(8,2) = k;
          }
        }
        dtcool   = std::min(dtcool,dttemp); // for next, not current timestep. Hence, has to be aggressive.
      }
    }
  }
  dens.DeleteAthenaArray();
  temp0.DeleteAthenaArray();

  return dtcool;
}

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
//========================================================================================
void UpdateGridData(Mesh *pm) {
  Real xMax;
  Real xMin;
  if (COORDINATE_SYSTEM == "cartesian") {
    xMax = pm->mesh_size.x1max;
    xMin = pm->mesh_size.x1min;
    pm->GridData(3) = xMax;
    pm->GridData(0) = xMin;

    xMax = pm->mesh_size.x2max;  
    xMin = pm->mesh_size.x2min;
    pm->GridData(7) = xMax;
    pm->GridData(4) = xMin;

    xMax = pm->mesh_size.x3max;
    xMin = pm->mesh_size.x3min;
    pm->GridData(11) = xMax;
    pm->GridData(8) = xMin;

    MeshBlock *pmb = pm->pblock;
    Real myVel = 0.0;
    Real pos  = 0.0;
    Real cellsize = 2.0*pm->mesh_size.x1max/pm->mesh_size.nx1;
    Real posUp = 0.9*(pm->mesh_size.x1max);// - 3.0*cellsize;
    Real posLow = 0.6*(pm->mesh_size.x1max);// - 15.0*cellsize;

    Real velAve = 0.0;
    Real vol = 0.0;   
    while (pmb != NULL) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          pos = std::sqrt( SQR( pmb->pcoord->x1v(i)) + SQR(pmb->pcoord->x2v(j))+ SQR(pmb->pcoord->x3v(k)));
          if ((pos<=posUp) && (pos>=posLow)) {
            velAve += 5.0*std::sqrt(  SQR(pmb->phydro->u(IM1,k,j,i)) 
                                    + SQR(pmb->phydro->u(IM2,k,j,i))
                                    + SQR(pmb->phydro->u(IM3,k,j,i)))
                         /  pmb->phydro->u(IDN,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
            vol += pmb->pcoord->GetCellVolume(k,j,i);
          }                  
        }
      }
    }
    pmb = pmb->next;
    }

#ifdef MPI_PARALLEL
    Real my_avg[2], avg[2];
    my_avg[0] = velAve;
    my_avg[1] = vol;
    MPI_Allreduce(&my_avg,&avg,2,MPI_ATHENA_REAL,MPI_SUM,
                  MPI_COMM_WORLD);
    velAve = avg[0];
    vol    = avg[1];
#endif
    myVel = velAve/vol*(pm->GridData(3)/(posLow));
    if ((myVel <=0.0)) {
      myVel = 0.0;
    }
    pm->GridData(2) = myVel;
    pm->GridData(6) = myVel;
    pm->GridData(10) = myVel;

  } else {
    xMax = pm->mesh_size.x1max;
    pm->GridData(3) = xMax;
    MeshBlock *pmb = pm->pblock;
    Real myVel = 0.0;
    Real pos  = 0.0;
    Real cellsize = pm->mesh_size.x1max/pm->mesh_size.nx1;
    Real posUp = 0.9*pm->mesh_size.x1max;
    Real posLow = 0.5*pm->mesh_size.x1max;// - 15.0*cellsize;
    Real velMax = 0.0;
    Real pVelMax = 0.0;   
    while (pmb != NULL) {
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            pos = pmb->pcoord->x1v(i);
            if ((pos<=posUp) && (pos>=posLow)) {
  
              velMax = std::max( pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i), velMax);
              
              pVelMax = std::max(std::sqrt( pmb->phydro->u(IEN,k,j,i)/pmb->phydro->u(IDN,k,j,i)), velMax);
              //velAve += pmb->phydro->u(IM1,k,j,i)/pmb->phydro->u(IDN,k,j,i)*pmb->pcoord->GetCellVolume(k,j,i);
              //vol += pmb->pcoord->GetCellVolume(k,j,i);
            }
                        
          }
        }
      }
    pmb = pmb->next;
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE,&velMax,1,MPI_ATHENA_REAL,MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&pVelMax,1,MPI_ATHENA_REAL,MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    
    myVel = std::max(velMax,pVelMax)*(pm->GridData(3)/(posLow));
    //std::cout << myVel << std::endl;
    if ((myVel <=0.0)) {
      myVel = 0.0;
    }
    pm->GridData(2) = 3.0*myVel;
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
    
    Real rout = pin->GetReal("problem","radius");
    Real rin  = rout - pin->GetOrAddReal("problem","ramp",0.0);

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

  icool = pin->GetOrAddInteger("problem","icool",0);
  if (icool > 0) {
    gm1      = pin->GetReal("hydro","gamma")-1.0;
    // coolsafe lowers the cooling (and thus overall) timestep. For
    // cooling curves without equilibrium, must be < 1. For
    // cooling curves with equilibrium, can be > 1. In that case,
    // the cooling timescale will generally not be resolved.
    coolsafe = pin->GetOrAddReal("problem","coolsafe",0.05);
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  }

  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = 4.0*PI;
    Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);

    SetMeanDensity(0.0);
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
  
  if (iturb == 2) {
    for (int k=ks; k<=ke; ++k) {
      Real x32 = SQR(pco->x3v(k)-x3fil);
      for (int j=js; j<=je; ++j) {
        Real x2 = fabs(pco->x2v(j))-x2fil;
        for (int i=1; i<=ngh; ++i) {
          Real x12= SQR(pco->x1v(ie+i)-x1fil);
          Real r  = std::sqrt(x12+x32);
          Real d  = damb+(dfil-damb)*0.25*(1.0-std::tanh((r -rfil)/0.05))
                                         *(1.0-std::tanh((x2-lfil)/0.05));
          prim(IDN,k,j,ie+i) = d;
          prim(IPR,k,j,ie+i) = pamb;
          prim(IVX,k,j,ie+i) = 0.0;
          prim(IVY,k,j,ie+i) = 0.0;
          prim(IVZ,k,j,ie+i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,j,ie+i) = 0.0;
            prim(NHYDRO-NSCALARS+1,k,j,ie+i) = 0.0;
            prim(NHYDRO-NSCALARS+2,k,j,ie+i) = d;
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          prim(IDN,k,j,ie+i) = damb;
          prim(IPR,k,j,ie+i) = pamb;  
          prim(IVX,k,j,ie+i) = 0.0;
          prim(IVY,k,j,ie+i) = 0.0;
          prim(IVZ,k,j,ie+i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,j,ie+i) = 0.0;
            prim(NHYDRO-NSCALARS+1,k,j,ie+i) = 0.0;
            prim(NHYDRO-NSCALARS+2,k,j,ie+i) = damb;
          }
        }
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(ie+i)) = bx0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(ie+i)) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(ie+i)) = bz0;
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

  if (iturb == 2) { // filament in y-z plane
    for (int k=ks; k<=ke; ++k) {
      Real x32 = SQR(pco->x3v(k)-x3fil);
      for (int j=1; j<=ngh; ++j) {
        Real x2 = fabs(pco->x2v(je+j))-x2fil;
        for (int i=is; i<=ie; ++i) {
          Real x12= SQR(pco->x1v(i)-x1fil);
          Real r  = std::sqrt(x12+x32);
          Real d  = damb+(dfil-damb)*0.25*(1.0-std::tanh((r -rfil)/0.05))
                                         *(1.0-std::tanh((x2-lfil)/0.05));
          prim(IDN,k,je+j,i) = d;
          prim(IPR,k,je+j,i) = pamb;
          prim(IVX,k,je+j,i) = 0.0;
          prim(IVY,k,je+j,i) = 0.0;
          prim(IVZ,k,je+j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,je+j,i) = 0.0;
            prim(NHYDRO-NSCALARS+1,k,je+j,i) = 0.0;
            prim(NHYDRO-NSCALARS+2,k,je+j,i) = d;
          }
        }
      }
    } 
  } else if (iturb == 1) { // blob at y > 0
    for (int k=ks; k<=ke; ++k) {
      Real x32 = SQR(pco->x3v(k)-x3cld);
      for (int i=is; i<=ie; ++i) {
        Real x12 = SQR(pco->x1v(i)-x1cld);
        for (int j=1; j<=ngh; ++j) {
          Real x22 = SQR(pco->x2v(je+j)-x2cld);
          Real r   = std::sqrt(x12+x22+x32);
          Real d             = damb+(dcld-damb)*0.5*(1.0-std::tanh((r-rcld)/0.05));
          prim(IDN,k,je+j,i) = d;
          prim(IPR,k,je+j,i) = pamb;
          prim(IVX,k,je+j,i) = 0.0;
          prim(IVY,k,je+j,i) = 0.0;
          prim(IVZ,k,je+j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,je+j,i) = 0.0;
            prim(NHYDRO-NSCALARS+1,k,je+j,i) = 0.0;
            prim(NHYDRO-NSCALARS+2,k,je+j,i) = d;
          }
        }
      }
    }
  } else if (iturb == 0) { // standard branch (no blob)
    for (int k=ks; k<=ke; ++k) {
      for (int i=is; i<=ie; ++i) {
#pragma omp simd
        for (int j=1; j<=ngh; ++j) {
          prim(IDN,k,je+j,i) = damb;
          prim(IPR,k,je+j,i) = pamb;  
          prim(IVX,k,je+j,i) = 0.0;
          prim(IVY,k,je+j,i) = 0.0;
          prim(IVZ,k,je+j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,je+j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+1,k,je+j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+2,k,je+j,i) = damb;
          }
        }
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int i=is; i<=ie+1; ++i) {
#pragma omp simd
        for (int j=1; j<=ngh; ++j) {
          b.x1f(k,je+j,i) = bx0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int i=is; i<=ie; ++i) {
#pragma omp simd
        for (int j=1; j<=ngh; ++j) {
          b.x2f(k,je+j,i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int i=is; i<=ie; ++i) {
#pragma omp simd
        for (int j=1; j<=ngh; ++j) {
          b.x3f(k,je+j,i) =  bz0;
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
  
  if (iturb == 2) {
    for (int k=1; k<=ngh; ++k) {
      Real x32 = SQR(pco->x3v(ke+k)-x3fil);
      for (int j=js; j<=je; ++j) {
        Real x2 = fabs(pco->x2v(j))-x2fil;
        for (int i=is; i<=ie; ++i) {
          Real x12= SQR(pco->x1v(i)-x1fil);
          Real r  = std::sqrt(x12+x32);
          Real d  = damb+(dfil-damb)*0.25*(1.0-std::tanh((r -rfil)/0.05))
                                         *(1.0-std::tanh((x2-lfil)/0.05));
          prim(IDN,ke+k,j,i) = d;
          prim(IPR,ke+k,j,i) = pamb;
          prim(IVX,ke+k,j,i) = 0.0;
          prim(IVY,ke+k,j,i) = 0.0;
          prim(IVZ,ke+k,j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,ke+k,j,i) = 0.0;
            prim(NHYDRO-NSCALARS+1,ke+k,j,i) = 0.0;
            prim(NHYDRO-NSCALARS+2,ke+k,j,i) = d;
          }
        }
      }
    }
  } else {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
#pragma omp simd
        for (int k=1; k<=ngh; ++k) {
          prim(IDN,ke+k,j,i) = damb;
          prim(IPR,ke+k,j,i) = pamb;  
          prim(IVX,ke+k,j,i) = 0.0;
          prim(IVY,ke+k,j,i) = 0.0;
          prim(IVZ,ke+k,j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,ke+k,j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+1,ke+k,j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+2,ke+k,j,i) = damb;
          }
        }
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
#pragma omp simd
        for (int k=1; k<=ngh; ++k) {
          b.x1f(ke+k,j,i) = bx0;  
        }
      }
    }
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
#pragma omp simd
        for (int k=1; k<=ngh; ++k) {
          b.x2f(ke+k,j,i) = by0;
        }
      }
    }
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
#pragma omp simd
        for (int k=1; k<=ngh; ++k) {
          b.x3f(ke+k,j,i) = bz0;
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
  
  if (iturb == 2) {
    for (int k=ks; k<=ke; ++k) {
      Real x32 = SQR(pco->x3v(k)-x3fil);
      for (int j=js; j<=je; ++j) {
        Real x2 = fabs(pco->x2v(j))-x2fil;
        for (int i=1; i<=ngh; ++i) {
          Real x12= SQR(pco->x1v(is-i)-x1fil);
          Real r  = std::sqrt(x12+x32);
          Real d  = damb+(dfil-damb)*0.25*(1.0-std::tanh((r -rfil)/0.05))
                                         *(1.0-std::tanh((x2-lfil)/0.05));
          prim(IDN,k,j,is-i) = d;
          prim(IPR,k,j,is-i) = pamb;
          prim(IVX,k,j,is-i) = 0.0;
          prim(IVY,k,j,is-i) = 0.0;
          prim(IVZ,k,j,is-i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,j,is-i) = 0.0;
            prim(NHYDRO-NSCALARS+1,k,j,is-i) = 0.0;
            prim(NHYDRO-NSCALARS+2,k,j,is-i) = d;
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          prim(IDN,k,j,is-i) = damb;
          prim(IPR,k,j,is-i) = pamb;  
          prim(IVX,k,j,is-i) = 0.0;
          prim(IVY,k,j,is-i) = 0.0;
          prim(IVZ,k,j,is-i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,j,is-i) = 0.0; 
            prim(NHYDRO-NSCALARS+1,k,j,is-i) = 0.0; 
            prim(NHYDRO-NSCALARS+2,k,j,is-i) = damb;
          }
        }
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = bx0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) = bz0;
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
  
  if (iturb == 2) {
    for (int k=ks; k<=ke; ++k) {
      Real x32 = SQR(pco->x3v(k)-x3fil);
      for (int j=1; j<=ngh; ++j) {
        Real x2 = fabs(pco->x2v(js-j))-x2fil;
        for (int i=is; i<=ie; ++i) {
          Real x12= SQR(pco->x1v(i)-x1fil);
          Real r  = std::sqrt(x12+x32);
          Real d  = damb+(dfil-damb)*0.25*(1.0-std::tanh((r -rfil)/0.05))
                                         *(1.0-std::tanh((x2-lfil)/0.05));
          prim(IDN,k,js-j,i) = d;
          prim(IPR,k,js-j,i) = pamb;
          prim(IVX,k,js-j,i) = 0.0;
          prim(IVY,k,js-j,i) = 0.0;
          prim(IVZ,k,js-j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,js-j,i) = 0.0;
            prim(NHYDRO-NSCALARS+1,k,js-j,i) = 0.0;
            prim(NHYDRO-NSCALARS+2,k,js-j,i) = d;
          }
        }
      }
    }
  } else { // iturb == 0,1
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IDN,k,js-j,i) = damb; 
          prim(IPR,k,js-j,i) = pamb;  
          prim(IVX,k,js-j,i) = 0.0;
          prim(IVY,k,js-j,i) = 0.0;
          prim(IVZ,k,js-j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,k,js-j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+1,k,js-j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+2,k,js-j,i) = damb;
          }
        }
      }
    }
  }
  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,js-j,i) = bx0;  
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f(k,js-j,i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f(k,js-j,i) = bz0;
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
  
  if (iturb == 2) {
    for (int k=1; k<=ngh; ++k) {
      Real x32 = SQR(pco->x3v(ks-k)-x3fil);
      for (int j=js; j<=je; ++j) {
        Real x2 = fabs(pco->x2v(j))-x2fil;
        for (int i=is; i<=ie; ++i) {
          Real x12 = SQR(pco->x1v(i)-x1fil);
          Real r   = std::sqrt(x12+x32);
          Real d   = damb+(dfil-damb)*0.25*(1.0-std::tanh((r -rfil)/0.05))
                                          *(1.0-std::tanh((x2-lfil)/0.05));
          prim(IDN,ks-k,j,i) = d;
          prim(IPR,ks-k,j,i) = pamb;
          prim(IVX,ks-k,j,i) = 0.0;
          prim(IVY,ks-k,j,i) = 0.0;
          prim(IVZ,ks-k,j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,ks-k,j,i) = 0.0;
            prim(NHYDRO-NSCALARS+1,ks-k,j,i) = 0.0;
            prim(NHYDRO-NSCALARS+2,ks-k,j,i) = d;
          }
        }
      }
    }
  } else {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IDN,ks-k,j,i) = damb;
          prim(IPR,ks-k,j,i) = pamb;  
          prim(IVX,ks-k,j,i) = 0.0;
          prim(IVY,ks-k,j,i) = 0.0;
          prim(IVZ,ks-k,j,i) = 0.0;
          if (NSCALARS == 3) {
            prim(NHYDRO-NSCALARS  ,ks-k,j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+1,ks-k,j,i) = 0.0; 
            prim(NHYDRO-NSCALARS+2,ks-k,j,i) = damb;
          }
        }
      }
    }
  }

  // no magnetic fields in ambient medium
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=ke; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(ks-1,j,i) = bx0;  
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f(ks-k,j,i) = by0; 
        }
      }
    }
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f(ks-k,j,i) = bz0;
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

  int iblast;
  Real gamma, rout, dr, esne, msne, pint, dint, v0;

  // cooling
  CoolingFunc = NULL;
  icool    = pin->GetOrAddInteger("problem","icool",0); // 0: adiabatic, 1: Slyz, 2: SlyzMod
  if (icool == 1) {
    CoolingFunc = CoolingFuncSlyz;
  } else if (icool == 2) {
    CoolingFunc = CoolingFuncSlyzMod;
  }

  // Perturbations
  // 0 none; 1: blob at boundaries, 2: filament
  iturb    = pin->GetOrAddInteger("problem","iturb",0); 
  // 0 standard blast wave (constant overpressure), 1: knova-type (all kinetic energy)
  iblast   = pin->GetOrAddInteger("problem","iblast",0);
  if (iblast == 1) {
    v0     = pin->GetReal("problem","v0");
  }

  // shell parameters
  rout = pin->GetReal("problem","radius");
  dr   = pin->GetOrAddReal("problem","ramp",0.1);
  pamb = pin->GetReal("problem","pamb");
  damb = pin->GetReal("problem","damb");
  esne = pin->GetReal("problem","esne");
  msne = pin->GetReal("problem","msne");
  if (MAGNETIC_FIELDS_ENABLED) {
    bx0 = pin->GetReal("problem","bx0");
    by0 = pin->GetReal("problem","by0");
    bz0 = pin->GetReal("problem","bz0");
  }
  gamma = peos->GetGamma();
  gm1   = gamma - 1.0;
  dint  = 3.0*msne/(4.0*PI*SQR(rout)*rout);
  pint  = gm1*3.0*esne/(4.0*PI*SQR(rout)*rout);
  if (Globals::my_rank == 0) {
    std::cout << "[ProblemGenerator]: iturb  = " << std::setw(13) << iturb << std::endl;
    std::cout << "[ProblemGenerator]: iblast = " << std::setw(13) << iblast << std::endl;
    std::cout << "[ProblemGenerator]: dint   = " << std::scientific << std::setw(13) << std::setprecision(5) << dint << std::endl;
    std::cout << "[ProblemGenerator]: pint   = " << std::scientific << std::setw(13) << std::setprecision(5) << pint << std::endl;
  }
  
  if (iturb == 1) {
    x1cld = pin->GetOrAddReal("problem","x1cld",0.2);
    x2cld = pin->GetOrAddReal("problem","x2cld",2.0);
    x3cld = pin->GetOrAddReal("problem","x3cld",0.2);
    dcld  = pin->GetOrAddReal("problem","dcld",damb);
    rcld  = pin->GetOrAddReal("problem","rcld",0.0);
  }
  if (iturb == 2) {
    x1fil = pin->GetOrAddReal("problem","x1fil",0.0);
    x2fil = pin->GetOrAddReal("problem","x2fil",0.0);
    x3fil = pin->GetOrAddReal("problem","x3fil",0.0);
    dfil  = pin->GetOrAddReal("problem","dfil",damb);
    rfil  = pin->GetOrAddReal("problem","rfil",0.2);
    lfil  = pin->GetOrAddReal("problem","lfil",3.0);
  }

  if (NSCALARS > 0) {
    if (NSCALARS != 3) {
      std::stringstream msg;
      msg << "### FATAL ERROR in magshell ProblemGenerator" << std::endl
          << "NSCALARS must be 3, but is " << NSCALARS << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }

  // setup uniform ambient medium with spherical over-pressured region
  if (iturb == 2) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real x,y,z,rad,den,pres,radf,len;
          x                    = pcoord->x1v(i)-x1fil;
          y                    = pcoord->x2v(j)-x2fil;
          z                    = pcoord->x3v(k)-x3fil;
          rad                  = std::sqrt(SQR(x) + SQR(y) + SQR(z));
          radf                 = std::sqrt(SQR(x) + SQR(z));
          len                  = fabs(y)-lfil;
          den                  = damb + (dint-damb)* 0.5*(1.0-std::tanh((rad -rout)/dr))
                                      + (dfil-damb)*0.25*(1.0-std::tanh((radf-rfil)/0.05))
                                                        *(1.0-std::tanh((len -lfil)/0.05));
          phydro->u(IDN,k,j,i) = den;
          if (iblast == 1) {
            Real rxy    = std::sqrt(SQR(pcoord->x1v(i))+SQR(pcoord->x2v(j)));
            Real cosphi = pcoord->x1v(i)/rxy;
            Real sinphi = pcoord->x2v(j)/rxy;
            Real costhe = rxy/rad;
            Real sinthe = pcoord->x3v(k)/rad;
            Real rprof  = (rad/rout) * 0.5*(1.0-std::tanh((rad -rout)/0.05)); 
            phydro->u(IM1,k,j,i) = v0 * rprof * costhe * cosphi * dint;
            phydro->u(IM2,k,j,i) = v0 * rprof * costhe * sinphi * dint;
            phydro->u(IM3,k,j,i) = v0 * rprof * sinthe * dint;
          } else {
            phydro->u(IM1,k,j,i) = 0.0;
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;
          }
          if (NON_BAROTROPIC_EOS) {
            pres                 = pamb + (pint-pamb)*0.5*(1.0-std::tanh((rad-rout)/dr));
            phydro->u(IEN,k,j,i) = pres/gm1 + 0.5*(  SQR(phydro->u(IM1,k,j,i))
                                                   + SQR(phydro->u(IM2,k,j,i))
                                                   + SQR(phydro->u(IM3,k,j,i)))
                                                 / phydro->u(IDN,k,j,i);
            if (DUAL_ENERGY)
              phydro->u(IIE,k,j,i) = pres/gm1;
          }

          if (NSCALARS == 3) {
            phydro->u(NHYDRO-NSCALARS  ,k,j,i) = dint*0.5*(1.0-std::tanh((rad-rout)/dr)); // ejecta 
            phydro->u(NHYDRO-NSCALARS+1,k,j,i) = phydro->u(NHYDRO-NSCALARS,k,j,i)*0.5*(1.0+std::tanh((rad-0.8*rout)/dr)); // shell
            phydro->u(NHYDRO-NSCALARS+2,k,j,i) = damb + (dfil-damb)*0.25*(1.0-std::tanh((radf-rfil)/0.05))
                                                                        *(1.0-std::tanh((len -lfil)/0.05));
 
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real x,y,z,rad,den,pres;
          x                    = pcoord->x1v(i);
          y                    = pcoord->x2v(j);
          z                    = pcoord->x3v(k);
          rad                  = std::sqrt(SQR(x) + SQR(y) + SQR(z));
          den                  = damb + (dint-damb)*0.5*(1.0-std::tanh((rad-rout)/dr));
          phydro->u(IDN,k,j,i) = den;
          if (iblast == 1) {
            Real rxy    = std::sqrt(SQR(pcoord->x1v(i))+SQR(pcoord->x2v(j)));
            Real cosphi = pcoord->x1v(i)/rxy;
            Real sinphi = pcoord->x2v(j)/rxy;
            Real costhe = rxy/rad;
            Real sinthe = pcoord->x3v(k)/rad;
            Real rprof  = (rad/rout) * 0.5*(1.0-std::tanh((rad -rout)/0.05));
            phydro->u(IM1,k,j,i) = v0 * rprof * costhe * cosphi * den;
            phydro->u(IM2,k,j,i) = v0 * rprof * costhe * sinphi * den;
            phydro->u(IM3,k,j,i) = v0 * rprof * sinthe * den;
          } else {
            phydro->u(IM1,k,j,i) = 0.0;
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;
          }
          if (NON_BAROTROPIC_EOS) {
            pres                 = pamb + (pint-pamb)*0.5*(1.0-std::tanh((rad-rout)/dr));
            phydro->u(IEN,k,j,i) = pres/gm1 + 0.5*(  SQR(phydro->u(IM1,k,j,i))
                                                   + SQR(phydro->u(IM2,k,j,i))
                                                   + SQR(phydro->u(IM3,k,j,i)))
                                                 / phydro->u(IDN,k,j,i);
            if (DUAL_ENERGY) 
              phydro->u(IIE,k,j,i) = pres/gm1;
          }

          if (NSCALARS == 3) {
            phydro->u(NHYDRO-NSCALARS  ,k,j,i) = dint*0.5*(1.0-std::tanh((rad-rout)/dr)); // ejecta 
            phydro->u(NHYDRO-NSCALARS+1,k,j,i) = phydro->u(NHYDRO-NSCALARS,k,j,i)*0.5*(1.0+std::tanh((rad-0.8*rout)/dr)); // shell
            phydro->u(NHYDRO-NSCALARS+2,k,j,i) = damb*0.5*(1.0+std::tanh((rad-rout)/dr)); // ambient
          }
        }
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = bx0;
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x2f(k,j,i) = by0;
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = bz0;
        }
      }
    }
    if (NON_BAROTROPIC_EOS) { // careful here: mix of volume and face positions...
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            phydro->u(IEN,k,j,i) += 0.5*(  SQR(pfield->b.x1f(k,j,i))
                                       + SQR(pfield->b.x2f(k,j,i))
                                       + SQR(pfield->b.x3f(k,j,i)));
          }
        }
      }
    }
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Called for individual meshblock
//========================================================================================

void MeshBlock::UserWorkInLoop(void) {
  return; 
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief Called for individual meshblock
//========================================================================================

void Mesh::UserWorkInLoop(void) {

  MeshBlock *pmb=pblock;

  const int nq = 7;
  Real qtot[nq]; // 0: vol, 1: dens, 2: vtot, 3: etot, 4: eint, 5: ekin, 6: emag
  Real qmin[nq];
  Real qmax[nq];
  Real ener[nq];
  Real trac[3];
  Real lengrat[2];
  for (int q=0; q<nq; q++) {
    qtot[q] = 0.0;
    qmin[q] = (FLT_MAX);
    qmax[q] = -(FLT_MAX);
    ener[q] = 0.0;
  }
  for (int q=0; q<3; q++) 
    trac[q] = 0.0;
  for (int q=0; q<2; q++)
    lengrat[q] = (FLT_MAX);
  Real u[NHYDRO];

  while (pmb != NULL) { // collect results from individual pmbs
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      Real x3  = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; j++) {
        Real x2  = pmb->pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; i++) {
          if (DUAL_ENERGY) {
            if (   isnan(pmb->phydro->u(IEN,k,j,i)) 
                || isnan(pmb->phydro->u(IDN,k,j,i))
                || isnan(pmb->phydro->u(IIE,k,j,i))
                || (pmb->phydro->u(IEN,k,j,i) <= 0.0) 
                || (pmb->phydro->u(IDN,k,j,i) <= 0.0)
                || (pmb->phydro->u(IIE,k,j,i) <= 0.0)) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
                        << " ei=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IIE,k,j,i)
                        << std::endl;
            } 
          } else {
            if (   isnan(pmb->phydro->u(IEN,k,j,i))
                || isnan(pmb->phydro->u(IDN,k,j,i))
                || (pmb->phydro->u(IEN,k,j,i) <= 0.0)
                || (pmb->phydro->u(IDN,k,j,i) <= 0.0)) {
              std::cout << "[UserWorkInLoop]: Warning: i=" << std::setw(4) << i << " j=" << std::setw(4) << j << " k=" << std::setw(4) << k
                        << " d =" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IDN,k,j,i)
                        << " m1=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM1,k,j,i)
                        << " m2=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM2,k,j,i)
                        << " m3=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IM3,k,j,i)
                        << " et=" << std::scientific << std::setw(11) << std::setprecision(3) << pmb->phydro->u(IEN,k,j,i)
                        << std::endl;
            }
          }
          Real dx1  = pmb->pcoord->dx1f(i);
          Real x1  = pmb->pcoord->x1v(i);
          Real dvol = pmb->pcoord->GetCellVolume(k,j,i);
          // energy
          for (int q=0; q<NHYDRO; q++) 
            u[q] = pmb->phydro->u(q,k,j,i);
          qtot[0] += dvol;
          ener[1]  = u[IDN];
          ener[2]  = u[IEN];
          ener[4]  = 0.5*(SQR(u[IM1])+SQR(u[IM2])+SQR(u[IM3]))/u[IDN];
          if (MAGNETIC_FIELDS_ENABLED) 
            ener[5] = 0.5*(  SQR(pmb->pfield->b.x1f(k,j,i))
                           + SQR(pmb->pfield->b.x2f(k,j,i)) 
                           + SQR(pmb->pfield->b.x3f(k,j,i)));
          if (DUAL_ENERGY) {
            ener[6] = u[IIE];
          }
          ener[3] = ener[2]-ener[4]-ener[5];
          for (int q=1; q<nq; q++) {
            qtot[q] += ener[q]*dvol;
            if (ener[q] < qmin[q]) qmin[q] = ener[q];
            if (ener[q] > qmax[q]) qmax[q] = ener[q];
          } 
          // shell tracking
          Real rad = std::sqrt(x1*x1+x2*x2+x3*x3);
          trac[0] += u[IDN];
          trac[1] += (u[IM1]*x1+u[IM2]*x2+u[IM3]*x3)/rad;
          trac[2] += u[IDN]*rad;
          Real temp = gm1*ener[2]/u[IDN];
          if (CoolingFunc != NULL) 
            lengrat[0] = std::min(lengrat[0],temp*std::sqrt(temp)/(dx1*fabs(CoolingFunc(u[IDN],temp)))); // cooling length
          lengrat[1] = std::min(lengrat[1],std::sqrt(PI*temp/u[IDN])/dx1); // Jeans length
        }
      }
    }
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  int ierr;
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qtot,nq,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmin,nq,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&qmax,nq,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&trac,3 ,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&lengrat,2,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif
  for (int q=1; q<nq; q++) qtot[q] /= qtot[0];
  for (int q=1; q<3;  q++) trac[q] /= trac[0];

  if (Globals::my_rank==0) {
    std::cout << "[UserWorkInLoop]: lcool= " << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[0]
              << " lgrv= "                   << std::scientific << std::setw(13) << std::setprecision(5) << lengrat[1]
              << std::endl;
    std::cout << "[UserWorkInLoop]: vrad = " << std::scientific << std::setw(13) << std::setprecision(5) << trac[1]
              << " rad = "                   << std::scientific << std::setw(13) << std::setprecision(5) << trac[2]
              << std::endl;
    std::cout << "[UserWorkInLoop]: dens = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[1]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[1]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[1]
              << std::endl;
    std::cout << "[UserWorkInLoop]: etot = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[2]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[2]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[2]
              << std::endl;
    std::cout << "[UserWorkInLoop]: eint0= " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[3]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[3]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[3]
              << std::endl;
    std::cout << "[UserWorkInLoop]: ekin = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[4]
              << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[4]
              << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[4]
              << std::endl;
    if (MAGNETIC_FIELDS_ENABLED) 
      std::cout << "[UserWorkInLoop]: emag = " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[5]
                << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[5]
                << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[5]
                << std::endl;
    if (DUAL_ENERGY)
      std::cout << "[UserWorkInLoop]: eint1= " << std::scientific << std::setw(13) << std::setprecision(5) << qtot[6]
                << " min = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmin[6]
                << " max = "                   << std::scientific << std::setw(13) << std::setprecision(5) << qmax[6]
                << std::endl;

  }
  
  if (qmin[3] <= 0.0) {
    std::cout << "[UserWorkInLoop]: eint < 0" << std::endl;
    stop_this();
  }

  return;
}
