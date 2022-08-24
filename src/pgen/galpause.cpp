//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file galpause.cpp
//  \brief Expanding Galactic wind to find galactopause
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
#include <map>
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

//typedef Real (*CoolingFunc_t)(const Real dens, const Real temp); // For generic cooling functions

class CoolingFunction;
CoolingFunction* pcoolfunc;

int nx1, icool, iprof;
Real gm1, temp0, fwind, mdot, vwind, gacc, csound2;
Real coolsafe = 0.05;
//Real CoolingFuncSlyz(const Real dens, const Real temp);
//Real CoolingFuncSlyzMod(const Real dens, const Real temp);
//Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt);
//Real BracketRoot(const Real dens, const Real temp0, const Real dt);
//Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt);
//Real GetEquiTemp(const Real dens,const Real temp0);
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
//CoolingFunc_t CoolingFunc;
Real HaloProfile(const Real r);
Real BetaProfile(const Real r); // Sets ambient density for beta model
Real IsoProfile(const Real r);
Real WindProfile(const Real r); // Sets wind model.
Real BetaPotential(const Real x1, const Real x2, const Real x3, const Real time);
Real LinearPotential(const Real x1, const Real x2, const Real x3, const Real time);

static void stop_this();

//========================================================================================
// Time Dependent Grid Functions
//  \brief Functions for time dependent grid, including two example boundary conditions
//========================================================================================
Real WallVel(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
void UpdateGridData(Mesh *pm);

//Global Variables for OuterX1
Real bx0,by0,bz0;

void OuterX1_Beta(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void InnerX1_Wind(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

Real PowerGridX1(Real x, RegionSize rs);

void ShockDetector(AthenaArray<Real> data, AthenaArray<Real> grid, int outArr[], Real eps);


//========================================================================================
// Unit mapping
//========================================================================================
std::map <std::string, Real> units;

//========================================================================================
// short for debugging interrupt
//========================================================================================
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//========================================================================================
// Returns the density for the beta model at beta = 0.5
//========================================================================================

Real HaloProfile(const Real r) {
  if (iprof == 0) {
    return IsoProfile(r);
  } else {
    return BetaProfile(r);
  }
}

Real BetaProfile(const Real r) {
  const Real a   = 2.82e-2;
  const Real b   = 0.5;
  const Real kpc = 8.80388e-3;
  const Real n00 = 1.0;
  Real rc = pow(a/n00,1.0/(3.0*b)); // in kpc
  Real nb = pow(n00*(1.0+SQR(r*kpc/rc)),-1.5*b);
  //fprintf(stdout,"[BetaProfile]: r=%13.5e n=%13.5e\n",r,nb);
  return nb;
}

Real IsoProfile(const Real r) {
  Real n = std::exp(-gacc*r/csound2);
  return n;
}

Real WindProfile(const Real r) {
  Real nw = mdot/(4.0*PI*fwind*SQR(r)*vwind);
  //fprintf(stdout,"[WindProfile]: r=%13.5e n=%13.5e\n",r,nw);
  return nw;
}

Real BetaPotential(const Real x1, const Real x2, const Real x3, const Real time) {
  const Real a   = 2.82e-2;
  const Real b   = 0.5;
  const Real kpc = 8.80388e-3;
  const Real n00 = 1.0;
  Real rc  = pow(a/n00,1.0/(3.0*b)); // in kpc
  Real phi = 1.5*b*temp0*std::log(SQR(rc/kpc)+SQR(x1));
  return phi;
}

Real LinearPotential(const Real x1, const Real x2, const Real x3, const Real time) {
  return x1*gacc;
}

//========================================================================================
// Class CoolingFunction
// see pyyacc.py
//========================================================================================
class CoolingFunction {

  private:
    const int nccoeff      = 13;
    const int nhcoeff      = 3;
    const Real fac         = 2.167177868e+31;
    const Real trangec[13] = {3.0,  5e1,  1e3,   8e3,  2.0e4,  4e4,    8e4, 1.13e5, 1.33e5,   1.86e5,   6.6e5,1.0e6,  1.33e7};
    const Real cexpo[13]   = {3.0,  1.1,  0.1,   5.0,    0.8, -1.0,   -2.0,   -3.0,   -1.8,    -0.09,  4.8965,  3.5,     3.5};
    const Real loss0       = 4.0e-30;
    const Real gain0       = 1.2e-24;
    const Real trangeh[3]  = {3.0,5e3,1e10};
    const Real hexpo[3]    = {0.0,-1.0,0.0};
    Real lhcoeff[3];
    Real ltrangeh[3];
    Real lccoeff[13];
    Real ltrangec[13];

  public:
    CoolingFunction() {
      lccoeff[0] = std::log(loss0);
      for (int i=0; i<nccoeff; ++i) ltrangec[i] = std::log(trangec[i]);
      for (int i=1; i<nccoeff; ++i) lccoeff[i]  = lccoeff[i-1] + cexpo[i-1]*(ltrangec[i]-ltrangec[i-1]);
      lhcoeff[0] = std::log(gain0);
      for (int i=0; i<nhcoeff; ++i) ltrangeh[i] = std::log(trangeh[i]);
      for (int i=1; i<nhcoeff; ++i) lhcoeff[i]  = lhcoeff[i-1] + hexpo[i-1]*(ltrangeh[i]-ltrangeh[i-1]);
      return;
    };
 
    ~CoolingFunction () {
      return;
    };

    Real EvalLoss(const Real temp) {
      Real lt = std::log(temp);
      if (lt < ltrangec[0]) {
        return 0.0;
      } else {
        int i=0;
        while ((i < nccoeff) && (ltrangec[i] <= lt)) i++;
        return std::exp(lccoeff[i-1]+cexpo[i-1]*(lt-ltrangec[i-1]));
      }
    };

    Real EvalGain(const Real temp) {
      Real lt = std::log(temp);
      if (lt < ltrangeh[0]) {
        return std::exp(lhcoeff[0]);
      } else {
        int i=0;
        while ((i < nhcoeff) && (ltrangeh[i] <= lt)) i++;
        return std::exp(lhcoeff[i-1]+hexpo[i-1]*(lt-ltrangeh[i-1]));
      }
    };

    Real HeatCoolFunc(const Real dens, const Real temp) {
      Real gamma  = EvalGain(temp); 
      Real lambda = EvalLoss(temp); 
      Real dedt = (gamma-dens*lambda)*fac;
      return dedt;
    };

    //========================================================================================
    // Real RootFunc(const Real dens, const Real temp0, const Real, temp1, const Real dt)
    // This is not the thermal equilibrium, but the implicit update for the thermal ODE
    //   dT = dt*(gamma-1)/kB * (Gamma(T)-n*Lambda(T)).
    //   As long as dt is limited to a fraction of dtcool (see HeatCool), the implicit solution
    //   prevents under- or over-shoots.
    //========================================================================================
    Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt) {
      return temp0 + dt*gm1*HeatCoolFunc(dens,temp1) - temp1;
    };

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
    };

    //========================================================================================
    // Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt)
    // \brief Finds root for RootFunc via bisection. Version without if-statements.
    //========================================================================================
    Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt) {
      if (HeatCoolFunc(dens,temp0) == 0.0) return temp0; // Nothing to do for thermal equilibrium
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
    };

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
      L[0] = HeatCoolFunc(dens,temp0);
      if (L[0] == 0.0) return temp0;
      // bracket
      sig  = SIGN(L[0]); 
      fac  = 1.0 + sig*0.1;
      T[0] = temp0;
      T[1] = fac*T[0];
      L[1] = HeatCoolFunc(dens,T[1]);
      while (L[0]*L[1] > 0.0) {
        T[0] = T[1];
        T[1] = fac*T[0];
        L[0] = L[1];
        L[1] = HeatCoolFunc(dens,T[1]);
      }
      // root
      nit = (int) (log(fabs(T[1]-T[0])/tol)/log(2.0));
      T[2] = 0.5*(T[0]+T[1]);
      L[0] = HeatCoolFunc(dens,T[0]);
      L[1] = HeatCoolFunc(dens,T[2]);
      for (int i=0; i<nit; i++) {
        int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
        T[w]  = T[2];
        L[w]  = L[1];
        T[2]  = 0.5*(T[0]+T[1]);
        L[1]  = HeatCoolFunc(dens,T[2]);
      }
      return T[2];
    };
}; // class CoolingFunction

//========================================================================================
//! \fn void heatcool(...)
//  \brief Heating and cooling for user-defined cooling function
//  Implicit solution of ODE for temperature change (see RootFunc)
//  (See HeatCoolTimeStep).
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
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dens(i)  = prim(IDN,k,j,i); 
        temp0(i) = prim(IGE,k,j,i)/dens(i); // IGE is pressure
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
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        temp1(i) = pcoolfunc->BracketRoot(dens(i),temp0(i),dt);
        temp2(i) = pcoolfunc->FindRoot(dens(i),temp0(i),temp1(i),dt);
      }
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dener(i)         = dens(i)*(temp2(i)-temp0(i))/g1;
        cons(IEN,k,j,i) += dener(i);
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
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        dens(i)  = w(IDN,k,j,i); 
        temp0(i) = w(IGE,k,j,i)/dens(i); // IGE is pressure
      } 
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dttemp   = coolsafe*temp0(i)/(fabs(pcoolfunc->HeatCoolFunc(dens(i),temp0(i)))+1e-60);
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
  if (dir != gridData(1)){
    retVal = 0.0;
  } else if (myX<=gridData(0)){
    retVal = 0.0;
  } else if (myX > gridData(0)){ 
    if (gridData(2)==0.0) retVal = 0.0;
    else retVal = gridData(2) * (myX-gridData(0))/(gridData(3)-gridData(0));
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

  if (COORDINATE_SYSTEM != "spherical_polar") {
    std::stringstream msg;
    msg << "### FATAL ERROR in galpause.cpp: coordinate system must be spherical-polar" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
      
  if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) 
    EnrollUserBoundaryFunction(INNER_X1,InnerX1_Wind);
  if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) 
    EnrollUserBoundaryFunction(OUTER_X1,OuterX1_Beta);

  Real x1rat = pin->GetOrAddReal("mesh","x1rat",1.0);
  nx1        = pin->GetInteger("mesh","nx1");
  if (x1rat < 0.0) 
    EnrollUserMeshGenerator(X1DIR,PowerGridX1);

  if (EXPANDING_ENABLED) {
    SetGridData(4);
    EnrollGridDiffEq(WallVel);
    EnrollCalcGridData(UpdateGridData);
    
    Real rout = pin->GetReal("problem","radius");
    Real rin  = rout - pin->GetOrAddReal("problem","ramp",0.0);

    GridData(0) = mesh_size.x1min;
    GridData(1) = 1; 
    GridData(2) = 0.0;
    GridData(3) = mesh_size.x1max; 
  }

  gm1      = pin->GetReal("hydro","gamma")-1.0;
  icool    = pin->GetOrAddReal("problem","icool",0);
  coolsafe = pin->GetOrAddReal("problem","coolsafe",0.05);
  if (icool == 1) {
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  }

  iprof   = pin->GetOrAddReal("problem","iprof",0); // 0: isothermal, 1: beta-model
  if (iprof == 0) {
    EnrollStaticGravPotFunction(LinearPotential);
  } else if (iprof == 1) {
    EnrollStaticGravPotFunction(BetaPotential);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in galpause.cpp: invalid iprof" << std::endl;
    throw std::runtime_error(msg.str().c_str());
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
void OuterX1_Beta(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        Real r = pco->x1v(i);
        Real d = HaloProfile(r);
        prim(IDN,k,j,ie+i) = d;
        prim(IPR,k,j,ie+i) = d*temp0;  
        prim(IGE,k,j,ie+i) = d*temp0;
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
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
//! \fn void InnerX1_UniformMedium(MeshBlock *pmb, Coordinates *pco, 
//                                 AthenaArray<Real> &prim,FaceField &b, Real time,
//                                 Real dt, int is, int ie, int js, int je,
//                                 int ks, int ke, int ngh) {
//  \brief Function for inner boundary being a uniform medium with density, velocity,
//   and pressure given by the global variables listed at the beginning of the file.
//========================================================================================

void InnerX1_Wind(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        Real r = pco->x1v(i);
        Real d = WindProfile(r);
        prim(IDN,k,j,is-i) = d;
        prim(IPR,k,j,is-i) = d*temp0;  
        prim(IGE,k,j,is-i) = d*temp0;
        prim(IVX,k,j,is-i) = vwind;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
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
//! \fn Real PowerGridX1(Real x, RegionSize rs)
//  \brief Generates grid following r_i = r_0*(1+delta)**i
//========================================================================================
Real PowerGridX1(Real x, RegionSize rs) {
  Real delta = pow((rs.x1max/rs.x1min),1.0/((Real)nx1))-1.0;
  Real r     = rs.x1min*pow((1.0+delta),x*((Real)nx1));
  return r;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  Real gamma, rwind, dr;

  // cooling
  pcoolfunc = new CoolingFunction();

  iprof   = pin->GetOrAddReal("problem","iprof",0); // 0: isothermal, 1: beta-model
  icool   = pin->GetOrAddReal("problem","icool",0); // 0: no cooling, 1: cooling
  // wind parameters
  rwind   = pin->GetReal("problem","radius");
  dr      = pin->GetOrAddReal("problem","ramp",0.1);
  fwind   = pin->GetOrAddReal("problem","fwind",0.1);
  vwind   = pin->GetReal("problem","vwind");
  mdot    = pin->GetReal("problem","mdot");
  temp0   = pin->GetReal("problem","temp0");
  gamma   = peos->GetGamma();
  gm1     = gamma - 1.0;
  csound2 = temp0;
  gacc    = pin->GetReal("problem","gacc");
  
  // only spherical coordinates
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real r     = pcoord->x1v(i); 
        Real dhalo = HaloProfile(r);
        Real dwind = WindProfile(r);
        Real den   = dhalo + (dwind-dhalo)*0.5*(1.0-std::tanh((r-rwind)/dr));
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = vwind*den*0.5*(1.0-std::tanh((r-rwind)/dr));
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = den*temp0/gm1 + 0.5*(  SQR(phydro->u(IM1,k,j,i))
                                                      + SQR(phydro->u(IM2,k,j,i))
                                                      + SQR(phydro->u(IM3,k,j,i)))
                                                    / phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = den*temp0/gm1;
        }
        fprintf(stdout,"[galpause]: i=%5i r=%13.5e n=%13.5e m1=%13.5e p=%13.5e e=%13.5e\n",
                i,pcoord->x1v(i),phydro->u(IDN,k,j,i),phydro->u(IM1,k,j,i),den*temp0,phydro->u(IEN,k,j,i));
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
          lengrat[0] = std::min(lengrat[0],temp*std::sqrt(temp)/(dx1*fabs(pcoolfunc->HeatCoolFunc(u[IDN],temp)))); // cooling length
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
