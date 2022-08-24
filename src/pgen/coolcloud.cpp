//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file coolcloud.cpp
//  \brief Problem generator for coolcloud: precipitation
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

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#if (NSCALARS != 2)
#error: coolcloud requires NSCALARS==2
#endif


// Cooling variables. These need to be set in InitUserMeshData (restarts!!)
Real n0, T0, gm1, grav_acc, pcool, dtcool = HUGE_NUMBER;
int64_t rseed; // seed for turbulence power spectrum
AthenaArray<Real> dvturb;

typedef Real (*CoolingFunc_t)(const Real dens, const Real temp);

//====================================================================================
// local functions
void InitTurbulence(ParameterInput *pin, Coordinates *pcoord, Hydro *phydro);
Real CoolingFuncShull(const Real dens, const Real temp); // heating and cooling
Real CoolingFuncSlyz(const Real dens, const Real temp);
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt);
Real BracketRoot(const Real dens, const Real temp0, const Real dt);
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt);
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);
void ProjectPressureInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh);
void ProjectPressureOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                            FaceField &b, Real time, Real dt,
                            int is, int ie, int js, int je, int ks, int ke, int ngh);
CoolingFunc_t CoolingFunc;

static void stop_this();

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//====================================================================================
// Real CoolingFuncShull(const Real dens, const Real temp)
//   cooling function from Shull & Moss 2020.
//   Returns de/dt [erg/s] in code units assuming ISM units with n0=T0=1.
//   Modified to provide constant temperature in center ("exclusion") region.
Real CoolingFuncShull(const Real dens, const Real temp) {
  const Real Lambda0 = 2e-22;
  const Real fac = 2.167177868e+31;
  const Real nenH = 1.165/SQR(2.247);
  const Real Tref = 1e6;
  const Real Ts   = 1e5;
  // from Shull & Moss
  Real Lambda = Lambda0 * std::pow(temp/Tref,-0.7);
  //Real dedt   = -dens*nenH*Lambda*fac;
  // modified Shull & Moss to allow for equilibrium (cutoff at 10^5)
  const Real Gamma0 = n0*Lambda0*std::pow(Ts/Tref,1.3); 
  Real Gamma = Gamma0*std::pow(Tref/temp,3.0);
  Real dedt = nenH*(Gamma-dens*Lambda)*fac;
  return dedt;
}

//====================================================================================
// Real CoolingFuncSlyz(const Real dens, const Real temp)
//   Cooling function from Slyz et al. 2005 (piece-wise powerlaw fit)
//   Returns de/dt [ergs/s] in code units assuming ISM units with n0=T0=1.
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

//====================================================================================
// Real RootFunc(const Real dens, const Real temp0, const Real, temp1, const Real dt)
// This is not the thermal equilibrium, but the implicit update for the thermal ODE
//   dT = dt*(gamma-1)/kB * (Gamma(T)-n*Lambda(T))
//
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real dt) {
  return temp0 + dt*gm1*CoolingFunc(dens,temp1) - temp1;
}

//====================================================================================
// Real BracketRoot(const Real temp0, const Real dt)
Real BracketRoot(const Real dens, const Real temp0, const Real dt) {
  Real rf    = RootFunc(dens,temp0,temp0,dt);
  Real sig   = (Real) ((rf > 0) - (rf < 0));
  Real fac   = 1.0 + sig*0.1;
  Real temp1 = temp0;
  while (rf*RootFunc(dens,temp0,temp1,dt) > 0)
    temp1 *= fac;
  return temp1;
}

//====================================================================================
// Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt)
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

//====================================================================================
// Real InitTurbulence()
void InitTurbulence(ParameterInput *pin, Coordinates *pcoord, Hydro *phydro) {
  MeshBlock* pmb = phydro->pmy_block;
  std::stringstream msg;
#ifdef MPI_PARALLEL
  int  mpierr;
#endif
  int64_t iseed;
  int kturblo, kturbhi, nkturb, nx1, nx2, nx3;
  Real pturb,vturb;
  Real x1len,x2len,x3len, vol, gvol;
  Real vrms[3], gvrms[3], cvrms[3], gcvrms[3], vavg[3], gvavg[3], cvavg[3], gcvavg[3], vrms3, cvrms3;
  AthenaArray<Real> phases;
  
  nx1     = pmb->ie-pmb->is+1;
  nx2     = pmb->je-pmb->js+1;
  nx3     = pmb->ke-pmb->ks+1;
  x1len   = pin->GetReal("mesh","x1max")-pin->GetReal("mesh","x1min");
  x2len   = pin->GetReal("mesh","x2max")-pin->GetReal("mesh","x2min");
  x3len   = pin->GetReal("mesh","x3max")-pin->GetReal("mesh","x3min");
  vol     = x1len*x2len*x3len;
  kturblo = pin->GetOrAddInteger("problem","kturblo",1);
  kturbhi = pin->GetOrAddInteger("problem","kturbhi",16);
  pturb   = pin->GetOrAddReal("problem","pturb",-2.0);
  vturb   = pin->GetReal("problem","vturb");
  iseed   = (int64_t) pin->GetOrAddInteger("problem","iseed",-151);
  nkturb  = kturbhi-kturblo+1;
  dvturb.NewAthenaArray(3,nx3,nx2,nx1);
  phases.NewAthenaArray(3,3,nkturb,nkturb,nkturb);

  // set the phases
  for (int c=0; c<3; c++) {
    for (int a=0; a<3; a++) {
      for (int k=0; k<nkturb; k++) {
        for (int j=0; j<nkturb; j++) {
          for (int i=0; i<nkturb; i++) {
            phases(c,a,k,j,i) = 2.0*PI*ran2(&iseed);
          }
        }
      }
    }
  }
  
  // determine mean (bulk) velocities
  for (int c=0; c<3; c++) {
    vavg[c]  = 0.0;
    gvavg[c] = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      Real z  = pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real x    = pcoord->x1v(i);
          Real dvol = pcoord->GetCellVolume(k,j,i);
          dvturb(c,k,j,i) = 0.0;
          for (int kk=0; kk<nkturb; kk++) {
            Real kz = 2.0*PI*((Real) (kk+kturblo))/x3len;
            for (int jj=0; jj<nkturb; jj++) {
              Real ky = 2.0*PI*((Real) (jj+kturblo))/x2len;
              for (int ii=0; ii<nkturb; ii++) {
                Real kx = 2.0*PI*((Real) (ii+kturblo))/x1len;
                dvturb(c,k,j,i) +=  std::sin(kx*x+phases(c,0,kk,jj,ii))
                                   *std::sin(ky*y+phases(c,1,kk,jj,ii))
                                   *std::sin(kz*z+phases(c,2,kk,jj,ii))
                                   *std::pow(std::sqrt(SQR(kx)+SQR(ky)+SQR(kz)),pturb);
              }                
            }
          }
          vavg[c] += dvturb(c,k,j,i)*dvol;
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  mpierr = MPI_Allreduce(&vavg, &gvavg, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[InitTurbulence]: MPI_Allreduce error = " << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int c=0; c<3; c++) vavg[c]  = gvavg[c];
#endif // MPI_PARALLEL
  for (int c=0; c<3; c++) vavg[c] /= vol;
  if (Globals::my_rank == 0) {
    std::cout << "[InitTurbulence]: "
              << " vavg[0]=" << std::scientific << std::setprecision(5) << vavg[0] 
              << " vavg[1]=" << std::scientific << std::setprecision(5) << vavg[1] 
              << " vavg[2]=" << std::scientific << std::setprecision(5) << vavg[2] << std::endl; 
  }

  // amplitude of un-normalized velocity field
  for (int c=0; c<3; c++) {
    vrms[c]  = 0.0;
    gvrms[c] = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real dvol  = pcoord->GetCellVolume(k,j,i);
          vrms[c]   += SQR((dvturb(c,k,j,i)-vavg[c]))*dvol;     
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  mpierr = MPI_Allreduce(&vrms, &gvrms, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[InitTurbulence]: MPI_Allreduce error = " << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int c=0; c<3; c++) vrms[c]  = gvrms[c];
#endif // MPI_PARALLEL
  for (int c=0; c<3; c++) vrms[c] /= vol;
  vrms3 = std::sqrt((vrms[0]+vrms[1]+vrms[2])/3.0); 
  for (int c=0; c<3; c++) vrms[c]  = std::sqrt(vrms[c]);
  if (Globals::my_rank == 0) {
    std::cout << "[InitTurbulence]: "
              << " vrms[0]=" << std::scientific << std::setprecision(5) << vrms[0]
              << " vrms[1]=" << std::scientific << std::setprecision(5) << vrms[1]
              << " vrms[2]=" << std::scientific << std::setprecision(5) << vrms[2] 
              << " vrms3  =" << std::scientific << std::setprecision(5) << vrms3 << std::endl;
  }

  // normalize the amplitudes and check
  for (int c=0; c<3; c++) {
    cvrms[c]  = 0.0;
    gcvrms[c] = 0.0;
    cvavg[c]  = 0.0;
    gcvavg[c] = 0.0;
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real dvol        = pcoord->GetCellVolume(k,j,i);
          dvturb(c,k,j,i)  = (dvturb(c,k,j,i)-vavg[c])*vturb/vrms3;
          cvavg[c]        += dvturb(c,k,j,i)*dvol;
          cvrms[c]        += SQR(dvturb(c,k,j,i))*dvol;
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  mpierr = MPI_Allreduce(&cvrms, &gcvrms, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[InitTurbulence]: MPI_Allreduce error = " << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  mpierr = MPI_Allreduce(&cvavg, &gcvavg, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[InitTurbulence]: MPI_Allreduce error = " << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int c=0; c<3; c++) {
    cvavg[c]  = gcvavg[c];
    cvrms[c]  = gcvrms[c];
  }
#endif // MPI_PARALLEL
  cvrms3 = std::sqrt((cvrms[0]+cvrms[1]+cvrms[2])/(3.0*vol));
  for (int c=0; c<3; c++) {
    cvavg[c] /= vol;
    cvrms[c]  = std::sqrt(cvrms[c]/vol);
  }

  if (Globals::my_rank == 0) {
    std::cout << "[InitTurbulence]: "
              << " cvavg[0]=" << std::scientific << std::setprecision(5) << cvavg[0]
              << " cvavg[1]=" << std::scientific << std::setprecision(5) << cvavg[1]
              << " cvavg[2]=" << std::scientific << std::setprecision(5) << cvavg[2] << std::endl;
    std::cout << "[InitTurbulence]: "
              << " cvrms[0]=" << std::scientific << std::setprecision(5) << cvrms[0]
              << " cvrms[1]=" << std::scientific << std::setprecision(5) << cvrms[1]
              << " cvrms[2]=" << std::scientific << std::setprecision(5) << cvrms[2]
              << " cvrms3  =" << std::scientific << std::setprecision(5) << cvrms3 << std::endl;
  }
  phases.DeleteAthenaArray();
  
  return;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {

  if (!DUAL_ENERGY) {
    std::stringstream msg;
    msg << "[InitUserMeshData]: ieqos = 2 requires DUAL_ENERGY" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  gm1   = pin->GetReal("hydro","gamma")-1.0;
  pcool = pin->GetReal("problem","pcool");
  EnrollUserExplicitSourceFunction(HeatCool);
  EnrollUserTimeStepFunction(HeatCoolTimeStep);

  grav_acc = pin->GetOrAddReal("hydro","grav_acc3",0.0);
  if (grav_acc != 0.0) {
    EnrollUserBoundaryFunction(INNER_X3, ProjectPressureInnerX3);
    EnrollUserBoundaryFunction(OUTER_X3, ProjectPressureOuterX3);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Setup for shell sweep-up. Assumes shell is located at center of box, with
//  coordinates running from -L to L.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int iturb, iprob,icool;
  Real x0, y0, z0, r0, nrat, dtc = HUGE_NUMBER;
  std::stringstream msg;
  Real avg[2], my_avg[2];
#ifdef MPI_PARALLEL
  int mpierr, myid = Globals::my_rank;
  Real my_dtc;
#endif

  iturb    = pin->GetInteger("problem","iturb"); // 0: no turbulence, 1: turbulence
  iprob    = pin->GetInteger("problem","iprob"); // 0: constant density, 1: gaussian perturbation
  icool    = pin->GetInteger("problem","icool"); // 0: Shull \& Moss, 1: Slyz
  n0       = pin->GetOrAddReal("problem","n0",1.0); // background density
  T0       = pin->GetOrAddReal("problem","T0",1.0); // background temperature
  x0       = pin->GetOrAddReal("problem","x0",0.0); // x-center of cloud
  y0       = pin->GetOrAddReal("problem","y0",0.0); // y-center of cloud
  z0       = pin->GetOrAddReal("problem","z0",0.0); // z-center of cloud
  nrat     = pin->GetOrAddReal("problem","nrat",1.0); // amplitude of density perturbation
  if (iprob > 0) {
    r0       = pin->GetReal("problem","r0"); // cloud radius
  }

  // Set the cooling function
  if (icool == 0) {
    CoolingFunc = CoolingFuncShull; 
  } else if (icool == 1) {
    CoolingFunc = CoolingFuncSlyz;
  } else {
    msg << "[coolcloud]: icool must have values 0 or 1. " << icool << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // Generate the normalized velocity amplitudes 
  if (iturb > 0) {
    InitTurbulence(pin,pcoord,phydro);
  }
  
  // Periodic box with constant density and turbulent velocity field
  if (iprob == 0) { 
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IDN,k,j,i) = n0;
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          if (iturb == 1) {
            phydro->u(IM1,k,j,i) += dvturb(0,k,j,i)*phydro->u(IDN,k,j,i);
            phydro->u(IM2,k,j,i) += dvturb(1,k,j,i)*phydro->u(IDN,k,j,i);
            phydro->u(IM3,k,j,i) += dvturb(2,k,j,i)*phydro->u(IDN,k,j,i);
          }
          phydro->u(IEN,k,j,i) = n0*T0/gm1 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                                  +SQR(phydro->u(IM2,k,j,i))
                                                  +SQR(phydro->u(IM3,k,j,i)))
                                                /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) = n0*T0/gm1;
        }
      }
    } 
  }

  // periodic box with density profile and turbulent velocity field restricted to density
  // Isothermal halo at T0 if grav_acc != 0.
  if (iprob == 1) {
    grav_acc = phydro->psrc->GetG3();
    for (int l=0; l<2; l++) {
      avg[l]    = 0.0;
      my_avg[l] = 0.0;
    }
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k);
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          Real r = std::sqrt(SQR(x-x0)+SQR(y-y0)+SQR(z-z0));
          Real e = std::exp(grav_acc*z/T0);
          phydro->u(IDN,k,j,i) = n0*e+(nrat-1.0*e)*n0*0.5*(1.0-std::tanh((r-r0)/(0.1*r0)));
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          if (iturb == 1) {
            phydro->u(IM1,k,j,i) += dvturb(0,k,j,i)*phydro->u(IDN,k,j,i);
            phydro->u(IM2,k,j,i) += dvturb(1,k,j,i)*phydro->u(IDN,k,j,i);
            phydro->u(IM3,k,j,i) += dvturb(2,k,j,i)*phydro->u(IDN,k,j,i);
            avg[0]               += ( SQR(dvturb(0,k,j,i))
                                     +SQR(dvturb(1,k,j,i))
                                     +SQR(dvturb(2,k,j,i)))
                                   *phydro->u(IDN,k,j,i);
            avg[1]               += 0.5*(1.0-std::tanh((r-r0)/(0.1*r0)));
          }
          if (NSCALARS == 2) {
            // ns=0: cloud; ns=1: ambient
            phydro->u(NHYDRO-NSCALARS  ,k,j,i) = phydro->u(IDN,k,j,i)*0.5*(1.0-std::tanh((r-r0)/(0.1*r0)));
            phydro->u(NHYDRO-NSCALARS+1,k,j,i) = phydro->u(IDN,k,j,i)*0.5*(1.0+std::tanh((r-r0)/(0.1*r0)));
          }
        }
      }
    }
  
    if (iturb == 1) {
#ifdef MPI_PARALLEL
      for (int l=0; l<2; l++) my_avg[l] = avg[l];
      mpierr = MPI_Allreduce(&my_avg, &avg, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (mpierr) {
        msg << "[coolcloud]: MPI_Allreduce error = " << mpierr << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
#endif
      avg[0] /= avg[1];
    }

    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k);
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          Real r = std::sqrt(SQR(x-x0)+SQR(y-y0)+SQR(z-z0));
          Real e = std::exp(grav_acc*z/T0);
          phydro->u(IEN,k,j,i) =  n0*T0*e/gm1 //+0.5*(1.0+std::tanh((r-r0)/(0.1*r0)))*avg[0])/gm1 
                                 + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                        +SQR(phydro->u(IM2,k,j,i))
                                        +SQR(phydro->u(IM3,k,j,i)))
                                      /phydro->u(IDN,k,j,i);
          phydro->u(IIE,k,j,i) =  n0*T0*e/gm1; //+0.5*(1.0+std::tanh((r-r0)/(0.1*r0)))*avg[0])/gm1;
        }
      }
    }
  }

  // Set cooling time over all Meshblocks
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real dens0 = phydro->u(IDN,k,j,i);
        Real temp0 = phydro->u(IIE,k,j,i)*gm1/dens0;
        dtc = std::min(dtc,temp0/(gm1*(fabs(CoolingFunc(dens0,temp0))+1e-60)));
      }
    }
  }
#ifdef MPI_PARALLEL
  my_dtc = dtc;
  mpierr = MPI_Allreduce(&my_dtc, &dtc, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[coolcloud]: MPI_Allreduce error = " << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
#endif
  dtcool = dtc;
  if (myid == 0) {
    std::cout << "[coolcloud]: dtcool = " << std::scientific << std::setprecision(5) << dtcool << std::endl;
  }

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
//========================================================================================

void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  Real g1  = pmb->peos->GetGamma()-1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dens  = prim(IDN,k,j,i);
        Real temp0 = prim(IGE,k,j,i)/dens; 
        if (temp0 <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in coolcloud.cpp: HeatCool: temp0 <=0 for DUAL_ENERGY" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        Real temp1       = BracketRoot(dens,temp0,dt);
        Real temp2       = FindRoot(dens,temp0,temp1,dt);
        Real dener       = dens*(temp2-temp0);
        dtcool           = std::min(dtcool,temp0/(fabs(CoolingFunc(dens,temp0))+1e-60));
        cons(IEN,k,j,i) += dener/g1;
        cons(IIE,k,j,i) += dener/g1;
      }
    }
  }
  //fprintf(stdout,"[HeatCool]: time = %13.5e dt = %13.5e dtcool = %13.5e\n",time,dt,dtcool);
  return;
}

//========================================================================================
//! \fn void HeatCoolTimeStep(...)
//  \brief Calculates cooling timestep and sends it to new_blockdt
//    Geometric mean between cfl timestep and cooling time, dt = dtcfl^(1-p) * dtcool^p,
//    if dtcool < dtcfl.
//========================================================================================

Real HeatCoolTimeStep(MeshBlock *pmb)
{
  Real dt = pmb->pmy_mesh->dt;
  return dt*std::min(1.0,pow(dtcool/dt,pcool));
}

//----------------------------------------------------------------------------------------
//! \fn void ProjectPressureInnerX3()
//  \brief  Pressure is integated into ghost cells to improve hydrostatic eqm

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

