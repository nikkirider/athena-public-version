//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sweepup.cpp
//  \brief Problem generator for sweep-up shell
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
Real gm1, dstiff;
// Shell variables.
Real nstar,e51,n0,b0,ciso,rsh;
int64_t rseed; // seed for turbulence power spectrum
AthenaArray<Real> dvturb;

//====================================================================================
// local functions
void InitTurbulence(ParameterInput *pin, Coordinates *pcoord, Hydro *phydro);
Real GetL(const Real dens, const Real temp, const Real radi); // heating and cooling
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real radi, const Real dt);
Real BracketRoot(const Real dens, const Real temp0, const Real radi, const Real dt);
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real radi, const Real dt);
Real GetEquiTemp(const Real dens, const Real radi);
void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real HeatCoolTimeStep(MeshBlock *pmb);

static void stop_this();

Real rshell(Real t);
Real vshell(Real t);
Real nshell(Real t);
Real wshell(Real t);
Real tshell(Real r);
Real cshell(Real r);
Real rinner(Real r, Real t);
Real tinner(Real r, Real t);
Real vinner(Real r, Real t);

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//====================================================================================
// shell radius
Real rshell(Real t) {
  return (97.0/8.8) * pow(nstar*e51/n0,0.2) * pow(9.513*t,0.6);
}
Real vshell(Real t) {
  return (5.7/0.091) * pow(nstar*e51/n0,0.2) * pow(9.513*t,-0.4);
}
Real nshell(Real t) {
  return 32.0 * pow(nstar*e51,0.4) * pow(n0,0.6) * pow(9.513*t,-0.8) / SQR(ciso);
}
Real wshell(Real t) {
  Real ns = nshell(t);
  Real rs = rshell(t)*8.8;
  return (0.33333*(n0/ns)*rs + (7*1.98e33*0.9*nstar/(4*PI*1.66e-24*ns*3.08e18*SQR(3.08e18*rs))))/8.8;
}
Real tshell(Real r) {
  return pow(r/(97/8.8) * pow(nstar*e51/n0,-0.2),5.0/3.0)/9.513;
}
Real cshell(Real r) {
  return n0*r + (7.0/17.2)*0.9*nstar/(4*PI*SQR(r));
}
Real ninner(Real r, Real t) {
  Real rs = rshell(t);
  return 1.5e-3 * pow(nstar*e51,0.2) * std::sqrt(n0) * pow(1-r/rs,-0.8) * pow(9.513*t,-0.6);
}
Real tinner(Real r, Real t) {
  Real rs = rshell(t);
  return 1.1e6 * pow(nstar*e51,0.2) * pow(n0,0.1) * pow(1-r/rs,0.8) * pow(9.513*t,-0.2);
}
Real vinner(Real r, Real t) {
  Real rs = rshell(t);
  Real vs = vshell(t);
  return vs*std::exp(10*(r-rs)/rs);
}

//====================================================================================
// Real GetL(const Real dens, const Real temp)
//   cooling function assuming existence of thermal equilibrium
//   returns de/dt [erg/s] in code units assuming ISM units with n0=T0=1.
//   Modified to provide constant temperature in center ("exclusion") region.
Real GetL(const Real dens, const Real temp, const Real radi) {
  const Real fac = 2.1756879947982705e+29;
  const Real p0=2e-26,p1=1e7,p2=1.184e5,p3=1e3,p4=1.4e-2,p5=9.2e1,p6=0.0,p8=2.0; // modified KI 2002
  Real p7 = dstiff;
  //Real w    = pow(10,-5.0+2.5*(1.0+std::tanh((radi-0.9*rsh)/(0.1*rsh)))); // 0 for r < rsh, 1 for r > rsh
  //**//first try: Modify density to lower cooling. Leads to clover leaves. Increases cooling time.
  //Real dmod = (1.0-w)*1e-3 + w*dens; // switching between "HII density" and ambient (real) density
  //Real gain = p0*pow(1.0+dmod/p7,p8);
  //Real loss = p0*(p1*std::exp(-p2/(temp+p3))+p4*std::sqrt(temp)*std::exp(-p5/std::max(1.0,temp+p6)));
  //Real dedt = (gain-dmod*loss)*fac;
  //**//second try: set temperature
  Real w    = (1.0 - std::exp(-SQR(radi/0.2)) * ((Real) (nstar > 0.0)));
  Real gain = p0*pow(1.0+dens/p7,p8);
  Real loss = p0*(p1*std::exp(-p2/(temp+p3))+p4*std::sqrt(temp)*std::exp(-p5/std::max(1.0,temp+p6)));
  Real dedtf= (1.0+gm1)*(1e4-temp)/1e-6; // fast cooling time
  Real dedt = (1-w)*dedtf + w*(gain-dens*loss)*fac;
  return dedt;
}

//====================================================================================
// Real RootFunc(const Real dens, const Real temp0, const Real, temp1, const Real dt)
// This is not the thermal equilibrium, but the implicit update for the thermal ODE
//   dT = dt*(gamma-1)/kB * (Gamma(T)-n*Lambda(T))
//
Real RootFunc(const Real dens, const Real temp0, const Real temp1, const Real radi, const Real dt) {
  return temp0 + dt*gm1*GetL(dens,temp1,radi) - temp1;
}

//====================================================================================
// Real BracketRoot(const Real temp0, const Real dt)
Real BracketRoot(const Real dens, const Real temp0, const Real radi, const Real dt) {
  Real rf    = RootFunc(dens,temp0,temp0,radi,dt);
  Real sig   = (Real) ((rf > 0) - (rf < 0));
  Real fac   = 1.0 + sig*0.1;
  Real temp1 = temp0;
  while (rf*RootFunc(dens,temp0,temp1,radi,dt) > 0)
    temp1 *= fac;
  return temp1;
}

//====================================================================================
// Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real dt)
Real FindRoot(const Real dens, const Real temp0, const Real temp1, const Real radi, const Real dt) {
  if (GetL(dens,temp0,radi) == 0.0) return temp0; // Nothing to do for thermal equilibrium
  // Otherwise, temp1 and temp0 bracket the temperature down to which we should integrate.
  const Real tol = 1e-6;
  int nit = (int) (log(fabs(temp1-temp0)/tol)/log(2.0));
  Real T[3], L[2];
  T[0]         = temp0;
  T[1]         = temp1;
  T[2]         = 0.5*(T[0]+T[1]);
  L[0]         = RootFunc(dens,temp0,T[0],radi,dt);
  L[1]         = RootFunc(dens,temp0,T[2],radi,dt);
  for (int i=0; i<nit; i++) {
    int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
    T[w]  = T[2];
    L[w]  = L[1];
    T[2]  = 0.5*(T[0]+T[1]);
    L[1]  = RootFunc(dens,temp0,T[2],radi,dt);
  }
  return T[2];
}

//====================================================================================
// Real GetEquiTemp(const Real dens, const Real temp)
Real GetEquiTemp(const Real dens, const Real radi) {
  Real tt0=5.0,tt1=1e7;
  const Real tol = 1e-6;
  int nit = (int) (log(fabs(tt1-tt0)/tol)/log(2.0));
  Real T[3], L[2]; // T[0] is lower, T[1] is upper, T[2] is mid
  T[0]         = tt0;
  T[1]         = tt1;
  T[2]         = 0.5*(T[0]+T[1]);
  L[0]         = GetL(dens,T[0],radi);
  L[1]         = GetL(dens,T[2],radi);
  for (int i=0; i<nit; i++) {
    int w = (L[0]*L[1] < 0); // 0 if >0, 1 if <= 0
    T[w]  = T[2];
    L[w]  = L[1];
    T[2]  = 0.5*(T[0]+T[1]);
    L[1]  = GetL(dens,T[2],radi);
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
  kturbhi = pin->GetOrAddInteger("problem","kturbhi",32);
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
  if (SELF_GRAVITY_ENABLED) {
    Real four_pi_G = 48.0*std::atan(1.0); // ISM unit system (n,T) with G=1
    Real eps = 0.0;
    SetFourPiG(four_pi_G);
    SetGravityThreshold(eps);
    SetMeanDensity(0.0);
  }

  // enroll user-defined cooling function for ieqos == 2
  int ieqos = pin->GetInteger("problem","ieqos");
  if (ieqos == 2) {
    if (!NON_BAROTROPIC_EOS) {
      std::stringstream msg;
      msg << "[InitUserMeshData]: ieqos = 2 requires NON_BAROTROPIC_EOS" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    //if (!DUAL_ENERGY) {
    //  std::stringstream msg;
    //  msg << "[InitUserMeshData]: ieqos = 2 requires DUAL_ENERGY" << std::endl;
    //  throw std::runtime_error(msg.str().c_str());
    //}

    gm1   = pin->GetReal("hydro","gamma")-1.0;
    dstiff= pin->GetReal("problem","dstiff");
    EnrollUserExplicitSourceFunction(HeatCool);
    EnrollUserTimeStepFunction(HeatCoolTimeStep);
  } 
 
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Setup for shell sweep-up. Assumes shell is located at center of box, with
//  coordinates running from -L to L.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int ieqos,iturb,nr2;
  // variables for nstar>0
  Real rs,ts,vs,cs,dx1v,dx2v,dx3v,nbar,check,s,w;
  Real intf,dr,rmin,rmax;

  iturb    = pin->GetInteger("problem","iturb"); // 0: no turbulence, 1: turbulence
  ieqos    = pin->GetInteger("problem","ieqos"); // 0: isothermal, 1: adiabatic, 2: cooling
  if ((NON_BAROTROPIC_EOS) && (ieqos==0)) {
    std::stringstream msg;
    msg << "[sweepup]: ieqos = 0 requires barotropic eos" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (ieqos == 0) {
    ciso   = pin->GetReal("hydro","iso_sound_speed");
  }
  n0       = pin->GetOrAddReal("problem","d0",1.0); // background density
  e51      = pin->GetOrAddReal("problem","e51",1.0); 
  nstar    = pin->GetOrAddReal("problem","nstar",1.0);
  if (MAGNETIC_FIELDS_ENABLED) {
    b0 = pin->GetReal("problem","b0");
  }
  if (nstar > 0.0) {
    rs     = pin->GetReal("problem","rshell"); // initial shell radius. Provide this based on resolution.
    rsh    = rs; // global variable
    ts     = tshell(rs); // corresponding evolution time
    vs     = vshell(ts);
    if (Globals::my_rank == 0) {
      std::cout << "[ProblemGenerator]: " 
                << " n0="     << std::scientific << std::setprecision(5) << n0
                << " e51="    << std::scientific << std::setprecision(5) << e51
                << " nstar="  << std::scientific << std::setprecision(5) << nstar << std::endl
                << "[ProblemGenerator]: "
                << " rshell=" << std::scientific << std::setprecision(5) << rs
                << " tshell=" << std::scientific << std::setprecision(5) << ts
                << " vshell=" << std::scientific << std::setprecision(5) << vs
                << std::endl;
    }
    dx1v   = pcoord->dx1v(is); // assuming cartesian for now
    dx2v   = pcoord->dx2v(js);
    dx3v   = pcoord->dx3v(ks);

    // column density (mass) swept up by shell
    intf   = 0.0; 
    dr     = pcoord->dx1v(is);
    rmin   = 0.5*dr;
    rmax   = pin->GetReal("mesh","x1max")-0.5*dr;
    cs     = cshell(rs);
    w      = -20.0;
    s      = rs/5.0;
    nr2    = (pin->GetInteger("mesh","nx1"))/2;
    for (int i=0; i<nr2; i++) {
      Real r     = rmin + ((Real) i)*dr;
      // skewed gaussian, normalized except for width s. Integral will thus result in length s.
      Real f     = 0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI);
      intf      += f*dr;
    }
    nbar   = cs/intf;
    check  = 0.0;
    for (int i=0; i<nr2; i++) {
      Real r = rmin + ((Real) i)*dr;
      check += nbar*0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI);
    }
    check *= dr;
    if (Globals::my_rank == 0) {
      std::cout << "[ProblemGenerator]: " << " intf =" << std::scientific << std::setprecision(5) << intf
                                          << " nbar ="   << std::scientific << std::setprecision(5) << nbar
                                          << " cs = "   << std::scientific << std::setprecision(5) << cs
                                          << " check = "  << std::scientific << std::setprecision(5) << check
                                          << std::endl; 
    }
    if (Globals::my_rank == 0) {
      if (ieqos == 2) {
        for (int i=0; i<nr2; i++) {
          Real r    = rmin+((Real) i)*dr;
          Real dens = n0*0.5*(1.0+std::tanh(5.0*(r-rs+dr)/dr)) + nbar*0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI);
          Real temp = GetEquiTemp(dens,r);
          fprintf(stdout,"[ProblemGenerator]: i=%4i r=%13.5e f2=%13.5e dens=%13.5e temp=%13.5e prss=%13.5e vrad=%13.5e\n",
                  i,r,nbar*0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI),
                  dens,temp,dens*temp,vs*0.5*(1.0-std::tanh(2.0*(r-rs)/dx1v)) * vinner(r,ts)/vinner(rs,ts)); 
        }
      }
    }
  } // if (nstar > 0.0) 

  // Generate the turbulence
  if (iturb > 0) {
    InitTurbulence(pin,pcoord,phydro);
  }
  
  if (ieqos == 0) { //isothermal
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k);
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x    = pcoord->x1v(i);
          Real r    = std::sqrt(x*x+y*y+z*z);
          Real vrad            = vs*0.5*(1.0-std::tanh(2.0*(r-rs)/dx1v)) * vinner(r,ts)/vinner(rs,ts);
          Real d               = n0*0.5*(1.0+std::tanh(5.0*(r-rs+dr)/dr)) + nbar*0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI);
          phydro->u(IDN,k,j,i) = d;
          phydro->u(IM1,k,j,i) = (vrad*(x/r)+dvturb(0,k,j,i))*d;
          phydro->u(IM2,k,j,i) = (vrad*(y/r)+dvturb(1,k,j,i))*d;
          phydro->u(IM3,k,j,i) = (vrad*(z/r)+dvturb(2,k,j,i))*d;
        }
      }
    }
  } else if (ieqos == 2) { // adiabatic with cooling
    if (nstar > 0.0) {
      for (int k=ks; k<=ke; k++) {
        Real z = pcoord->x3v(k);
        for (int j=js; j<=je; j++) {
          Real y = pcoord->x2v(j);
          for (int i=is; i<=ie; i++) {
#ifdef SUBSAMPLE
            Real x = pcoord->x1v(i);
            int nq;
            if (DUAL_ENERGY) {
              nq = 6;
            } else {
              nq = 5;
            }
            int  nsub=4; // 6 fields, 4^3 subsampling points
            Real qsub[nq];     // hydro quantities IDN,IM1-3,IEN,IIE.
            for (int iq=0; iq<nq; iq++) {
              qsub[iq] = 0.0;
            }
            Real norm = 1.0/pow(((Real)nsub),3);
            Real d=0.0,m1=0.0,m2=0.0,m3=0.0,e=0.0,ei=0.0;
            for (int kk=0; kk<nsub; kk++) {
              Real zz = z-0.5*dx3v+(0.5+(Real)kk)*dx3v/((Real)nsub);
              for (int jj=0; jj<nsub; jj++) {
                Real yy = y-0.5*dx2v+(0.5+(Real)jj)*dx2v/((Real)nsub);
                for (int ii=0; ii<nsub; ii++) {
                  Real xx   = x-0.5*dx1v+(0.5+(Real)ii)*dx1v/((Real)nsub);
                  Real r    = std::sqrt(xx*xx+yy*yy+zz*zz); 
                  Real vrad = vs*0.5*(1.0-std::tanh(2.0*(r-rs)/dx1v)) * vinner(r,ts)/vinner(rs,ts);
                  Real dd   = n0*0.5*(1.0+std::tanh(5.0*(r-rs+dr)/dr)) + nbar*0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI);
                  qsub[0]   += dd;
                  Real mm1  = vrad*(xx/r)*dd;
                  Real mm2  = vrad*(yy/r)*dd;
                  Real mm3  = vrad*(zz/r)*dd;
                  Real tt   = GetEquiTemp(dd,r);
                  qsub[1]  += mm1;
                  qsub[2]  += mm2;
                  qsub[3]  += mm3;
                  qsub[4]  += dd*tt/gm1 + 0.5*(SQR(mm1)+SQR(mm2)+SQR(mm3))/dd;
                  if (DUAL_ENERGY)  
                    qsub[5]  += dd*tt/gm1;
                }
              }
            }
            for (int iq=0; iq<nq; iq++) {
              phydro->u(iq,k,j,i) = norm*qsub[iq];
            }
            if (iturb > 0) {
              for (int c=0; c<3; c++) {
                phydro->u(c+1,k,j,i) += dvturb(c,k,j,i)*phydro->u(IDN,k,j,i);
              }
              phydro->u(IEN,k,j,i) += 0.5*(SQR(dvturb(0,k,j,i))+SQR(dvturb(1,k,j,i))+SQR(dvturb(2,k,j,i)))*phydro->u(IDN,k,j,i);
            }
#else // SUBSAMPLE
            Real x               = pcoord->x1v(i);
            Real r               = std::sqrt(x*x+y*y+z*z);
            Real vrad            = vs*0.5*(1.0-std::tanh(2.0*(r-rs)/dx1v)) * vinner(r,ts)/vinner(rs,ts);
            Real d               = n0*0.5*(1.0+std::tanh(5.0*(r-rs+dr)/dr)) + nbar*0.5*std::exp(-0.5*SQR((r-rs)/s))*(1.0+std::erf(w*(r-rs)/std::sqrt(2.0)))/std::sqrt(2.0*PI);
            phydro->u(IDN,k,j,i) = d;
            phydro->u(IM1,k,j,i) = vrad*(x/r)*d;
            phydro->u(IM2,k,j,i) = vrad*(y/r)*d;
            phydro->u(IM3,k,j,i) = vrad*(z/r)*d;
            Real temp            = GetEquiTemp(d,r);
            //fprintf(stdout,"i,j,k=%4i%4i%4i dens=%13.5e temp=%13.5e\n",i,j,k,d,temp);
            phydro->u(IEN,k,j,i) = d*temp/gm1 + 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+SQR(phydro->u(IM3,k,j,i)))/d;
            if (DUAL_ENERGY)
              phydro->u(IIE,k,j,i) = d*temp/gm1;
            //fprintf(stdout,"[ProblemGenerator]: i,j,k=%4i%4i%4i d=%13.5e e=%13.5e ei=%13.5e m1=%13.5e m2=%13.5e m3=%13.5e temp=%13.5e\n",
            //        i,j,k,phydro->u(IDN,k,j,i),phydro->u(IEN,k,j,i),phydro->u(IIE,k,j,i),phydro->u(IM1,k,j,i),phydro->u(IM2,k,j,i),phydro->u(IM3,k,j,i),temp);
#endif // SUBSAMPLE
          }
        }
      }
    } else {// if (nstar > 0.0) 
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            phydro->u(IDN,k,j,i) = n0;
            if (iturb > 0) {
              phydro->u(IM1,k,j,i) = dvturb(0,k,j,i)*n0;
              phydro->u(IM2,k,j,i) = dvturb(1,k,j,i)*n0;
              phydro->u(IM3,k,j,i) = dvturb(2,k,j,i)*n0;
            } else {
              phydro->u(IM1,k,j,i) = 0.0;
              phydro->u(IM2,k,j,i) = 0.0;
              phydro->u(IM3,k,j,i) = 0.0;
            }
            Real temp            = GetEquiTemp(n0,100.0);
            phydro->u(IEN,k,j,i) = n0*temp/gm1 
                                  + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                         +SQR(phydro->u(IM2,k,j,i))
                                         +SQR(phydro->u(IM3,k,j,i)))
                                       /phydro->u(IDN,k,j,i);
            if (DUAL_ENERGY) 
              phydro->u(IIE,k,j,i) = n0*temp/gm1;
          }
        }
      }
    }
  } else {
    std::stringstream msg;
    msg << "[ProblemGenerator]: ieqos == 1 not implented yet" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
            pfield->b.x1f(k,j,i) = b0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x2f(k,j,i) = 0.0;
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          pfield->b.x3f(k,j,i) = 0.0;
        }
      }   
    }
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, 
                                       pcoord, is, ie, js, je, ks, ke);
    for (int k=ks; k <= ke; ++k) {
      for (int j=js; j <= je; ++j) {
        for (int i=is; i <= ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*( SQR(pfield->bcc(0,k,j,i))
                                       +SQR(pfield->bcc(1,k,j,i))
                                       +SQR(pfield->bcc(2,k,j,i)));
        }
      }
    }          
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief
//========================================================================================

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
//========================================================================================

void HeatCool(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  Real g1  = pmb->peos->GetGamma()-1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real z = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x     = pmb->pcoord->x1v(i);
        Real r     = std::sqrt(x*x+y*y+z*z);
        Real dens  = prim(IDN,k,j,i);
        Real temp0 = -1.0;
        if (DUAL_ENERGY) {
	  temp0 = prim(IGE,k,j,i)/dens; // primitive variables are just pressure
        } else {
          temp0 = prim(IPR,k,j,i)/dens;
        }
        if (temp0 <= 0.0) {
          std::stringstream msg;
          msg << "### FATAL ERROR in sweepup.cpp: HeatCool: temp0 <=0 for DUAL_ENERGY" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        Real temp1       = BracketRoot(dens,temp0,r,dt);
        Real temp2       = FindRoot(dens,temp0,temp1,r,dt);
        Real dener       = dens*(temp2 - temp0);
        cons(IEN,k,j,i) += dener/g1;
        if (DUAL_ENERGY)
          cons(IIE,k,j,i) += dener/g1;
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
  Real dtcool = HUGE_NUMBER;
  Real g1  = pmb->peos->GetGamma()-1.0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real z = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x     = pmb->pcoord->x1v(i);
        Real r     = std::sqrt(x*x+y*y+z*z);
        Real dens  = pmb->phydro->w(IDN,k,j,i);
        Real temp0 = 0.0;
        if (DUAL_ENERGY) {
          temp0 = pmb->phydro->w(IGE,k,j,i)/dens;
        } else {
          temp0 = pmb->phydro->w(IPR,k,j,i)/dens;
        }
        if (temp0 <= 0.0) { 
          std::stringstream msg;
          msg << "### FATAL ERROR in sweepup.cpp: HeatCool: temp0 <=0 for DUAL_ENERGY" << std::endl
              << "    p=" << std::setw(5) << Globals::my_rank << " i=" << std::setw(5) << i << " j=" << std::setw(5) << j << " k=" << std::setw(5) << k << std::endl
              << "    temp0=" << std::scientific << std::setw(13) << std::setprecision(5) << temp0
              << "    dens=" << std::scientific << std::setw(13) << std::setprecision(5) << dens << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
        dtcool = std::min(dtcool,temp0/(fabs(GetL(dens,temp0,r))+1e-60));
      }
    }
  }
  return dtcool; 
}


