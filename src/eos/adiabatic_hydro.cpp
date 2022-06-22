//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements functions in class EquationOfState for adiabatic hydrodynamics`

// C/C++ headers
#include <cmath>   // sqrt()
#include <cfloat>  // FLT_MIN

// Athena++ headers
#include "eos.hpp"
#include "../hydro/hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../field/field.hpp"

#define DEBUG_ALL
// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) {
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  gamma2_ = pin->GetReal("hydro", "gamma2");
  fprintf(stdout,"gamma=%13.5e,gamma2=%13.5e\n",gamma_,gamma2_);
  bgkc1_ = pin->GetOrAddReal("hydro","bgkc1",1e-2);
  bgkc2_ = pin->GetOrAddReal("hydro","bgkc2",2);
  density_floor_  = pin->GetOrAddReal("hydro","dfloor", std::sqrt(1024*(FLT_MIN)));
  pressure_floor_ = pin->GetOrAddReal("hydro","pfloor", std::sqrt(1024*(FLT_MIN)));
}

// destructor

EquationOfState::~EquationOfState() {
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//           const AthenaArray<Real> &prim_old, const FaceField &b,
//           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old, const FaceField &b, AthenaArray<Real> &prim,
  AthenaArray<Real> &bcc, Coordinates *pco, int il,int iu, int jl,int ju, int kl,int ku) {
  Real gm1 = GetGamma() - 1.0;

  fprintf(stdout,"FINDERScons il=%4i,jl=%4i,kl=%4i\n",il,jl,kl);
  fprintf(stdout,"FINDERScons iu=%4i,ju=%4i,ku=%4i\n",iu,ju,ku);

  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    if(fluidnum==1){
      gm1 = GetGamma2() - 1.0;
    }
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {

#ifdef DEBUG_ALL
          if(i==1||i==2||i==3||i==30||i==31||i==32||i==33||i==34||i==65||i==66){
            fprintf(stdout,"consdens=%11.3e,j=%4i,i=%4i\n",cons(fluidnum,IDN,k,j,i),j,i);
          }
#endif
//          fprintf(stdout,"consmom1=%11.3e,i=%4i\n",cons(0,1,k,j,i),i);
//          fprintf(stdout,"consmom2=%11.3e,i=%4i\n",cons(0,2,k,j,i),i);
//          fprintf(stdout,"consmom3=%11.3e,i=%4i\n",cons(0,3,k,j,i),i);
//          fprintf(stdout,"consenergy=%11.3e,i=%4i\n",cons(0,4,k,j,i),i);

          Real& u_d  = cons(fluidnum,IDN,k,j,i);
          Real& u_m1 = cons(fluidnum,IM1,k,j,i);
          Real& u_m2 = cons(fluidnum,IM2,k,j,i);
          Real& u_m3 = cons(fluidnum,IM3,k,j,i);
          Real& u_e  = cons(fluidnum,IEN,k,j,i);

          Real& w_d  = prim(fluidnum,IDN,k,j,i);
          Real& w_vx = prim(fluidnum,IVX,k,j,i);
          Real& w_vy = prim(fluidnum,IVY,k,j,i);
          Real& w_vz = prim(fluidnum,IVZ,k,j,i);
          Real& w_p  = prim(fluidnum,IPR,k,j,i);

        // apply density floor, without changing momentum or energy
          u_d = (u_d > density_floor_) ?  u_d : density_floor_;
          w_d = u_d;
      
          Real di = 1.0/u_d;
          w_vx = u_m1*di;
          w_vy = u_m2*di;
          w_vz = u_m3*di;
      
          Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
          w_p = gm1*(u_e - ke);
    
#ifdef DEBUG_ALL
          if(i==1||i==2||i==3||i==30||i==31||i==32||i==33||i==34||i==65||i==66){ 
            fprintf(stdout,"u_d=%11.3e,u_m1=%11.3e,u_e=%11.3e,w_d=%11.3e,w_vx=%11.3e,w_p=%11.3e\n",u_d,u_m1,u_e,w_d,w_vx,w_p);
          }
#endif 

//          fprintf(stdout,"prim(fluidnum,IPR,k,j,i)=%11.7e, w_p=%11.7e\n",prim(fluidnum,IPR,k,j,i),w_p);

          u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1) + ke);
          w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

//          fprintf(stdout,"prim(fluidnum,IPR,k,j,i)=%11.7e\n",prim(fluidnum,IPR,k,j,i));
        }
      }
     
    }
  }

  // passive scalars 
  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
  for (int n=(NHYDRO-NSCALARS); n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
              Real& u_s = cons(fluidnum,n  ,k,j,i);
              Real& u_d = cons(fluidnum,IDN,k,j,i);
              Real   di = 1./u_d;
              Real& w_s = prim(fluidnum,n,k,j,i);
              w_s = u_s*di;
            }
          }
        }
      } 
    }
  

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku);
// \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
     int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0/(GetGamma() - 1.0);

  fprintf(stdout,"FINDERS il=%4i,jl=%4i,kl=%4i\n",il,jl,kl);

  // Force outer-loop vectorization
  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    if(fluidnum==1){
      igm1 = 1.0/(GetGamma2() - 1.0);
    }
#pragma omp simd
  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
    //#pragma omp simd
#pragma novector
    for (int i=il; i<=iu; ++i) {

#ifdef DEBUG_ALL
          if(i==1||i==2||i==3||i==30||i==31||i==32||i==33||i==34||i==65||i==66){
            fprintf(stdout,"primdens=%11.3e,j=%4i,i=%4i\n",prim(0,IDN,k,j,i),j,i);
          }
#endif

          Real& u_d  = cons(fluidnum,IDN,k,j,i);
          Real& u_m1 = cons(fluidnum,IM1,k,j,i);
          Real& u_m2 = cons(fluidnum,IM2,k,j,i);
          Real& u_m3 = cons(fluidnum,IM3,k,j,i);
          Real& u_e  = cons(fluidnum,IEN,k,j,i);

          const Real& w_d  = prim(fluidnum,IDN,k,j,i);
          const Real& w_vx = prim(fluidnum,IVX,k,j,i);
          const Real& w_vy = prim(fluidnum,IVY,k,j,i);
          const Real& w_vz = prim(fluidnum,IVZ,k,j,i);
          const Real& w_p  = prim(fluidnum,IPR,k,j,i);

          u_d = w_d;
          u_m1 = w_vx*w_d;
          u_m2 = w_vy*w_d;
          u_m3 = w_vz*w_d;
          u_e = w_p*igm1 + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
 
#ifdef DEBUG_ALL
          if(i==1||i==2||i==3||i==30||i==31||i==32||i==33||i==34||i==65||i==66){
            fprintf(stdout,"u_d=%11.3e,u_m1=%11.3e,u_e=%11.3e,w_d=%11.3e,w_vx=%11.3e,w_p=%11.3e\n",u_d,u_m1,u_e,w_d,w_vx,w_p);
          }
#endif
        }
      }
    }
  }

  // passive scalars
  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){ 
  for (int n=(NHYDRO-NSCALARS); n<NHYDRO; ++n) { 
#pragma omp simd
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma novector
        for (int i=il; i<=iu; ++i) {
              Real& u_s = cons(fluidnum,n,k,j,i);
              const Real& w_s = prim(fluidnum,n  ,k,j,i);
              const Real& w_d = prim(fluidnum,IDN,k,j,i);
              u_s = w_s*w_d;
            }
         }
        }
      }
    }
  

  return;
}


//----------------------------------------------------------------------------------------
// \!fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
// \brief returns adiabatic sound speed given vector of primitive variables

Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]){
  fprintf(stdout,"sndspd=%11.5e\n",std::sqrt(gamma_*prim[IPR]/prim[IDN]));
  return std::sqrt(gamma_*prim[IPR]/prim[IDN]);
}



//---------------------------------------------------------------------------------------
Real EquationOfState::SoundSpeed2(const Real prim1[NHYDRO],const Real prim2[NHYDRO],const int fluidnum) {
    Real sndspd, T, K1, K2, eq9;
    K1=(5.0-3.0*gamma_)/(gamma_-1.0);
    K2=(5.0-3.0*gamma2_)/(gamma2_-1.0);
    eq9=0.25*((K1+3.0)*prim1[IDN] + (K2+3.0)*prim2[IDN])/((prim1[IPR]/(gamma_-1.0))+(prim2[IPR]/(gamma2_-1.0)));
    T=0.5/(eq9);
    sndspd=std::sqrt(T*gamma_);
    if(fluidnum==1){
      sndspd=std::sqrt(T*gamma2_);
    }
    fprintf(stdout,"EOS SOUNDSPEED K1=%11.3e,K2=%11.3e,eq9=%11.3e,T=%11.3e\n",K1,K2,eq9,T);
    fprintf(stdout,"sndspd=%11.5e\n",sndspd);   
    return sndspd;
}


//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states
void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int fluidnum, int k, int j, int i) {
      Real& w_d  = prim(fluidnum,IDN,k,j,i);
      Real& w_p  = prim(fluidnum,IPR,k,j,i);
      // apply density floor
      w_d = (w_d > density_floor_) ?  w_d : density_floor_;
      // apply pressure floor
      w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}
