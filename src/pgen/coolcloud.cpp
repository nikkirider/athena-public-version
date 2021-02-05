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

#define SUBSAMPLE


// Cooling variables. These need to be set in InitUserMeshData (restarts!!)
Real gm1, pcool, dtcool = HUGE_NUMBER;
int64_t rseed; // seed for turbulence power spectrum
AthenaArray<Real> dvturb;

//====================================================================================
// local functions
void InitTurbulence(ParameterInput *pin, Coordinates *pcoord, Hydro *phydro);
Real GetL(const Real dens, const Real temp, const Real radi); // heating and cooling
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);

static void stop_this();

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//====================================================================================
// Real GetL(const Real dens, const Real temp)
//   cooling function from Shull & Moss 2020.
//   Returns de/dt [erg/s] in code units assuming ISM units with n0=T0=1.
//   Modified to provide constant temperature in center ("exclusion") region.
Real GetL(const Real dens, const Real temp) {
  const Real Lambda0 = 2e-22;
  const Real fac = 2.167177868e+31;
  const Real nenH = 1.165/SQR(2.247);
  const Real T0  = 1e6;
  Real Lambda = Lambda0 * pow(temp/T0,-0.7);
  Real dedt   = -dens*nenH*Lambda*fac;
  return dedt;
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

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Setup for shell sweep-up. Assumes shell is located at center of box, with
//  coordinates running from -L to L.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int iturb, iprob;
  Real n0, T0, r0, nrat, dtc = HUGE_NUMBER;
  std::stringstream msg;
#ifdef MPI_PARALLEL
  int mpierr, myid = Globals::my_rank;
  Real my_dtc;
#endif

  iturb    = pin->GetInteger("problem","iturb"); // 0: no turbulence, 1: turbulence
  iprob    = pin->GetInteger("problem","iprob"); // 0: constant density, 1: gaussian perturbation
  n0       = pin->GetOrAddReal("problem","n0",1.0); // background density
  T0       = pin->GetOrAddReal("problem","T0",1.0); // background temperature
  nrat     = pin->GetOrAddReal("problem","nrat",1.0); // amplitude of density perturbation
  if (iprob > 0) {
    r0       = pin->GetReal("problem","r0"); // cloud radius
  }

  // Generate the turbulence
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

  // periodic box with density profile and turbulent velocity field
  if (iprob == 1) {
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k);
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          Real r = std::sqrt(x*x+y*y+z*z);
          phydro->u(IDN,k,j,i) = n0+(nrat-1.0)*n0*0.5*(1.0-std::tanh((r-r0)/(0.1*r0)));
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

  // Set cooling time over all Meshblocks
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real dens0 = phydro->u(IDN,k,j,i);
        Real temp0 = phydro->u(IIE,k,j,i)*gm1/dens0;
        dtc = std::min(dtc,temp0/(gm1*(fabs(GetL(dens0,temp0))+1e-60)));
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
//========================================================================================

void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  //if (dtcool == HUGE_NUMBER) // for the first iteration.
  //  dtcool = 0.25*dt;    // global variable for timestep, to be sent to HeatCoolTimeStep 
  //  dtcool = HUGE_NUMBER;
  Real g1  = pmb->peos->GetGamma()-1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real dens  = prim(IDN,k,j,i);
        Real temp0 = prim(IGE,k,j,i)*g1; // This should have gam-1, bc it's the specific internal energy
        if (temp0 <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in coolcloud.cpp: HeatCool: temp0 <=0 for DUAL_ENERGY" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        Real dedt        = GetL(dens,temp0);
        Real dener       = dt*dens*dedt;
        dtcool           = std::min(dtcool,temp0/(g1*fabs(dedt)+1e-60));
        cons(IEN,k,j,i) += dener;
        cons(IIE,k,j,i) += dener;
        //fprintf(stdout,"[HeatCool]: i,j,k=%4i%4i%4i ien=%13.5e iie=%13.5e dener=%13.5e dtcool=%13.5e\n",i,j,k,cons(IEN,k,j,i),cons(IIE,k,j,i),dener,dtcool);
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


