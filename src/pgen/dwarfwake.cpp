//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dwarfwake.cpp
//  \brief Problem generator for a dark matter sub-halo traveling through the MW gas halo
//

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <bits/stdc++.h>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../fft/athena_fft.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/utils.hpp"
// extra headers
#include "../hydro/srcterms/hydro_srcterms.hpp"


#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#if (NSCALARS != 2)
#error: coolcloud requires NSCALARS==2
#endif


// Cooling variables. These need to be set in InitUserMeshData (restarts!!)
Real n0, T0, gm1, v0, grav_acc, dtcool = HUGE_NUMBER;
const Real coolsafe = 0.05;

// Ahead declaration of class CoolingFunction
class CoolingFunction;
CoolingFunction* pcoolfunc;

//====================================================================================
// local functions
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh);
void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
Real gravpot_darkhalo(const Real x1, const Real x2, const Real x3, const Real time);
static void stop_this();

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//========================================================================================
// Class CoolingFunction
// see pyyacc.py
//========================================================================================
class CoolingFunction {

  private:
    int nccoeff = 0, nhcoeff = 0;
    const Real fac = 2.167177868e+31;
    bool useheat;
    Real gain0, loss0;
    AthenaArray<Real> lhcoeff, ltrangeh, trangeh, hexpo,
                      lccoeff, ltrangec, trangec, cexpo;

  public:

    // Constructor. Initializes lookup-table.
    CoolingFunction(int icool) {

      useheat = (icool > 0);

      if (std::fabs(icool) == 1) { // Cooling term slyzmod4
        nccoeff = 13;
        lccoeff.NewAthenaArray(nccoeff);
        ltrangec.NewAthenaArray(nccoeff);
        trangec.NewAthenaArray(nccoeff);
        cexpo.NewAthenaArray(nccoeff);
        const Real trc[13] = {3.0,  5e1,  1e3,   8e3,  2.0e4,  4e4,    8e4, 1.13e5, 1.33e5,   1.86e5,   6.6e5,1.0e6,  1.33e7};
        const Real cex[13] = {3.0,  1.1,  0.1,   5.0,    0.8, -1.0,   -2.0,   -3.0,   -1.8,    -0.09,  4.8965,  3.5,     3.5};
        loss0 = 4.0e-30;
        for (int i=0; i<nccoeff; ++i) {
          cexpo(i)   = cex[i]; 
          trangec(i) = trc[i];
        }
      } else if (std::fabs(icool) == 2) { // Standard Slyz
        nccoeff = 11;
        lccoeff.NewAthenaArray(nccoeff);
        ltrangec.NewAthenaArray(nccoeff);
        trangec.NewAthenaArray(nccoeff);
        cexpo.NewAthenaArray(nccoeff);
        const Real trc[11] = {3.0,  5e1,  1e3,   8e3,3.9811e4,  1e5,2.884e5,  4.732e5, 2.113e6,  3.981e6,  1.995e7};
        const Real cex[11] = {3.0,  1.1,  0.1, 2.867,     1.6, -0.2,   -3.0,    -0.22,    -3.0,     0.33,      0.5};
        loss0 = 4.0e-30;
        for (int i=0; i<nccoeff; ++i) {
          cexpo(i)   = cex[i];
          trangec(i) = trc[i];
        }
      } else if (std::fabs(icool) == 3) { // fake curve
        nccoeff = 6;
        lccoeff.NewAthenaArray(nccoeff);
        ltrangec.NewAthenaArray(nccoeff);
        trangec.NewAthenaArray(nccoeff);
        cexpo.NewAthenaArray(nccoeff);
        const Real trc[6] = {3.0,9.54992586e+01,8.91250938e+03,1.12201845e+04,8.91250938e+05,1.12201845e+06};
        const Real cex[6] = {20.0,0.4,20.0,0.4,20.0,0.4};
        loss0 = 1.0e-56;
        for (int i=0; i<nccoeff; ++i) {
          cexpo(i)   = cex[i];
          trangec(i) = trc[i];
        }
      }

      // heating branch
      if (icool == 1) { // Heating term slyzmod4
        nhcoeff = 3;
        lhcoeff.NewAthenaArray(nhcoeff);
        ltrangeh.NewAthenaArray(nhcoeff);
        trangeh.NewAthenaArray(nccoeff);
        hexpo.NewAthenaArray(nccoeff);
        const Real trh[3]  = {3.0,5e3,1e10};
        const Real hex[3]  = {0.0,-1.0,0.0};
        gain0 = 1.2e-24;
        for (int i=0; i<nhcoeff; ++i) {
          hexpo(i)   = hex[i];
          trangeh(i) = trh[i];
        }
      } else if (icool == 2) { // slyzmod1
        nhcoeff = 1;
        lhcoeff.NewAthenaArray(nhcoeff);
        ltrangeh.NewAthenaArray(nhcoeff);
        trangeh.NewAthenaArray(nccoeff);
        hexpo.NewAthenaArray(nccoeff);
        const Real trh[1]  = {1e10};
        const Real hex[1]  = {0.0};
        gain0 = 2.0e-25;
        for (int i=0; i<nhcoeff; ++i) {
          hexpo(i)   = hex[i];
          trangeh(i) = trh[i];
        }
      } else if (icool == 3) { // fake 
        nhcoeff = 1;
        lhcoeff.NewAthenaArray(nhcoeff);
        ltrangeh.NewAthenaArray(nhcoeff);
        trangeh.NewAthenaArray(nccoeff);
        hexpo.NewAthenaArray(nccoeff);
        const Real trh[1]  = {1e10};
        const Real hex[1]  = {0.0};
        gain0 = 1.0e-26;
        for (int i=0; i<nhcoeff; ++i) {
          hexpo(i)   = hex[i];
          trangeh(i) = trh[i];
        }
      }

      lccoeff(0) = std::log(loss0);
      for (int i=0; i<nccoeff; ++i) ltrangec(i) = std::log(trangec(i));
      for (int i=1; i<nccoeff; ++i) lccoeff(i)  = lccoeff(i-1) + cexpo(i-1)*(ltrangec(i)-ltrangec(i-1));
      if (useheat) {
        lhcoeff(0) = std::log(gain0);
        for (int i=0; i<nhcoeff; ++i) ltrangeh(i) = std::log(trangeh(i));
        for (int i=1; i<nhcoeff; ++i) lhcoeff(i)  = lhcoeff(i-1) + hexpo(i-1)*(ltrangeh(i)-ltrangeh(i-1));
      }
      return;
    };

    // Destructor
    ~CoolingFunction () {
      if (nccoeff > 0) {
        cexpo.DeleteAthenaArray();
        trangec.DeleteAthenaArray();
        lccoeff.DeleteAthenaArray();
        ltrangec.DeleteAthenaArray();
      }
      if (nhcoeff > 0) {
        hexpo.DeleteAthenaArray();
        trangeh.DeleteAthenaArray();
        lhcoeff.DeleteAthenaArray();
        ltrangeh.DeleteAthenaArray();
      }
      return;
    };

    Real EvalLoss(const Real temp) {
      Real lt = std::log(temp);
      if (lt < ltrangec(0)) {
        return 0.0;
      } else {
        int i=0;
        while ((i < nccoeff) && (ltrangec(i) <= lt)) i++;
        return std::exp(lccoeff(i-1)+cexpo(i-1)*(lt-ltrangec(i-1)));
      }
    };

    Real EvalGain(const Real temp) {
      if (useheat) {
        Real lt = std::log(temp);
        if (lt < ltrangeh(0)) {
          return std::exp(lhcoeff(0));
        } else {
          int i=0;
          while ((i < nhcoeff) && (ltrangeh(i) <= lt)) i++;
          return std::exp(lhcoeff(i-1)+hexpo(i-1)*(lt-ltrangeh(i-1)));
        }
      } else {
        return 0.0;
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
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {

  int icool, ipot;

  icool    = pin->GetInteger("problem","icool"); // 0: none , 1: Slyz
  ipot     = pin->GetInteger("problem","ipot"); // 0: none, 1: static dwarf potential
  gm1      = pin->GetReal("hydro","gamma")-1.0;
  grav_acc = pin->GetOrAddReal("hydro","grav_acc3",0.0);
  v0       = pin->GetOrAddReal("problem","v0",0.0);

  if (icool != 0) {
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  }
   
  if (ipot == 1) {
    EnrollStaticGravPotFunction(gravpot_darkhalo);
  }

  if (grav_acc != 0.0) {
    EnrollUserBoundaryFunction(INNER_X3, ProjectPressureInnerX3);
    EnrollUserBoundaryFunction(OUTER_X3, ProjectPressureOuterX3);
  }
  
  if (v0 != 0.0) {
    EnrollUserBoundaryFunction(INNER_X1, InflowBoundary);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Setup for dwarfwake problem.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int iprob,icool;
  Real x0, y0, z0, x1min, x1max;
  std::stringstream msg;
  Real avg[2], my_avg[2];

  iprob    = pin->GetInteger("problem","iprob"); // 0: constant density, 1: gaussian perturbation
  icool    = pin->GetInteger("problem","icool"); // 0: no cooling, 1: slyzmod4, 2: slyz. Negative icool: no heating
  n0       = pin->GetReal("problem","n0"); // background density
  T0       = pin->GetReal("problem","T0"); // background temperature
  v0       = pin->GetOrAddReal("problem","v0",0.0); // wind x-velocity
  x0       = pin->GetOrAddReal("problem","x0",0.0); // x-center of cloud
  y0       = pin->GetOrAddReal("problem","y0",0.0); // y-center of cloud
  z0       = pin->GetOrAddReal("problem","z0",0.0); // z-center of cloud
  x1min    = pin->GetReal("mesh","x1min"); 
  x1max    = pin->GetReal("mesh","x1max"); 

  // cooling function
  if (icool != 0) 
    pcoolfunc= new CoolingFunction(icool);

  // test for cooling function
  if (iprob == -1) { 
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k) ;
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          Real r = std::sqrt(SQR(x)+SQR(y)+SQR(z));
          phydro->u(IDN,k,j,i) = 1.07615e-4 + (1e-3-1.07615e-4)*0.5*(1.0-std::tanh((r-0.5*x1max)/(0.1*x1max)));
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) = 1.07615e-4*9.29239e5/gm1 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                                  +SQR(phydro->u(IM2,k,j,i))
                                                  +SQR(phydro->u(IM3,k,j,i)))
                                                /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = 1.07615e-4*9.29239e5/gm1;
        }
      }
    } 
  }

  // Constant density with x-velocity wind
  if (iprob == 0) { 
    if (icool > 0) {
      T0 = pcoolfunc->GetEquiTemp(n0,1e7);
      if (Globals::my_rank == 0)  
        fprintf(stdout,"[dwarfwake]: Reset T0 = %13.5e as thermal equilibrium temperature\n",T0);
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = n0;
          phydro->u(IM1,k,j,i) = v0*n0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) = n0*T0/gm1 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                                  +SQR(phydro->u(IM2,k,j,i))
                                                  +SQR(phydro->u(IM3,k,j,i)))
                                                /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = n0*T0/gm1;
        }
      }
    } 
  }

// Hydrostatic density profile within a dwarf satellite DM potential
// with a background x-velocity wind  ***EDIT THIS FUNCTION*** need tanh profile to make dwarf gas not move
  if (iprob == 1) {
    const Real a = 2.275e3; 
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k) ;
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          Real r = std::sqrt(SQR(x-x0)+SQR(y-y0)+SQR(z-z0));
          phydro->u(IDN,k,j,i) = 0.0858*std::exp(-7.448*(1-std::log(1+r/a)/(r/a)));
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*T0/gm1 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                                  +SQR(phydro->u(IM2,k,j,i))
                                                  +SQR(phydro->u(IM3,k,j,i)))
                                                /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = phydro->u(IDN,k,j,i)*T0/gm1;
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


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {

}

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
//// Inflow boundary condition
//// Inputs:
////   pmb: pointer to MeshBlock
////   pcoord: pointer to Coordinates
////   is,ie,js,je,ks,ke: indices demarkating active region
//// Outputs:
////   prim: primitives set in ghost zones
//========================================================================================

void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // Set hydro variables
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        prim(IDN,k,j,i) = n0;
        prim(IPR,k,j,i) = n0*T0;
        prim(IVX,k,j,i) = v0;
        prim(IVY,k,j,i) = 0.0;
        prim(IVZ,k,j,i) = 0.0;
        prim(IGE,k,j,i) = n0*T0;
      }
    }
  }
  return;
} 



//========================================================================================
//! \fn void ProjectPressureInnerX3()
//  \brief  Pressure is integated into ghost cells to improve hydrostatic eqm
//========================================================================================
//
//
void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      if (n==(IVZ)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IVZ,ks-k,j,i) = -prim(IVZ,ks+k-1,j,i);  // reflect 3-vel
        }
      } else if (n==(IPR)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IPR,ks-k,j,i) = prim(IPR,ks+k-1,j,i)
             - prim(IDN,ks+k-1,j,i)*grav_acc*(2*k-1)*pco->dx3f(k);
        }
      } else {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ks-k,j,i) = prim(n,ks+k-1,j,i);
        }
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) =  b.x1f((ks+k-1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) =  b.x2f((ks+k-1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = -b.x3f((ks+k  ),j,i);  // reflect 3-field
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureOuterX3()
//  \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      if (n==(IVZ)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IVZ,ke+k,j,i) = -prim(IVZ,ke-k+1,j,i);  // reflect 3-vel
        }
      } else if (n==(IPR)) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IPR,ke+k,j,i) = prim(IPR,ke-k+1,j,i)
             + prim(IDN,ke-k+1,j,i)*grav_acc*(2*k-1)*pco->dx3f(k);
        }
      } else {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ke+k,j,i) = prim(n,ke-k+1,j,i);
        }
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones, reflecting b3

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) =  b.x1f((ke-k+1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) =  b.x2f((ke-k+1),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = -b.x3f((ke-k+1),j,i);  // reflect 3-field
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn static gravpot_darkhalo
//  \brief spherical DM potential, centered at (0,0,0) 
//  ISM units
//----------------------------------------------------------------------------------------
static Real gravpot_darkhalo(const Real x1, const Real x2, const Real x3, const Real time){
  Real M200 = 1e10; //solar mass
  const Real h = 0.673; 
  Real logc200 = (0.905 - 0.101*log10(M200/(1e12/h))); 
  Real c200 = std::pow(10,logc200);
  Real delta_c = (200.0/3)*std::pow(c200,3)/(log(1+c200)-c200/(1+c200));
  const Real rho_crit = (1e-29)/(1.677e-24); // 10^29 g/cm^3 --> ISM units
  Real rho_s = rho_crit*delta_c; //  ISM density unit
  Real r200 = std::pow((M200/16.84)/(200.0*rho_crit*(4*M_PI/3)), 1.0/3); // ISM length unit
  Real rs = r200/c200; // ISM length unit
  Real r = std::sqrt(SQR(x1)+SQR(x2)+SQR(x3));
  Real Phi = (-4*M_PI*rho_s*std::pow(rs,3)*log(1+r/rs)/r);//*0.5*(1.0-std::tanh((r-r200)/(0.1*r200)));
  return Phi;
}


