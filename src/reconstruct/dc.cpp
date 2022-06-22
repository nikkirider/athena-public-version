//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dc.cpp
//  \brief piecewise constant (donor cell) reconstruction

// Athena++ headers
#include "reconstruction.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief

void Reconstruction::DonorCellX1(MeshBlock *pmb,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  // compute L/R states for each variable
  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
   for (int n=0; n<(NHYDRO); ++n) {
     for (int k=kl; k<=ku; ++k) {
     for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
            wl(fluidnum,n,k,j,i) = w(fluidnum,n,k,j,i-1);
            wr(fluidnum,n,k,j,i) = w(fluidnum,n,k,j,i  );

          }
        }
      }
    }}
  
  //fprintf(stdout,"axu1_f1=%13.5e,axu2_f1=%13.5e,axu1_f2=%13.5e,axu2_f2=%13.5e\n",wl(0,IVX,0,0,0),wr(0,IVX,0,0,0),wl(1,IVX,0,0,0),wr(1,IVX,0,0,0));
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        wl(IBY,k,j,i) = bcc(IB2,k,j,i-1);
        wr(IBY,k,j,i) = bcc(IB2,k,j,i  );
      }
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        wl(IBZ,k,j,i) = bcc(IB3,k,j,i-1);
        wr(IBZ,k,j,i) = bcc(IB3,k,j,i  );
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief

void Reconstruction::DonorCellX2(MeshBlock *pmb,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  // compute L/R states for each variable
  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        if(NFLUIDS==1){
            wl(fluidnum,n,k,j,i) = w(fluidnum,n,k,j-1,i);
            wr(fluidnum,n,k,j,i) = w(fluidnum,n,k,j  ,i);
          }
        }
      }
    }}
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        wl(IBY,k,j,i) = bcc(IB3,k,j-1,i);
        wr(IBY,k,j,i) = bcc(IB3,k,j  ,i);
      }
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        wl(IBZ,k,j,i) = bcc(IB1,k,j-1,i);
        wr(IBZ,k,j,i) = bcc(IB1,k,j  ,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief

void Reconstruction::DonorCellX3(MeshBlock *pmb,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  // compute L/R states for each variable
  for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
            wl(fluidnum,n,k,j,i) = w(fluidnum,n,k-1,j,i);
            wr(fluidnum,n,k,j,i) = w(fluidnum,n,k  ,j,i);
          }
        }
      }
    }}
  
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        wl(IBY,k,j,i) = bcc(IB1,k-1,j,i);
        wr(IBY,k,j,i) = bcc(IB1,k  ,j,i);
      }
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        wl(IBZ,k,j,i) = bcc(IB2,k-1,j,i);
        wr(IBZ,k,j,i) = bcc(IB2,k  ,j,i);
      }
    }}
  }

  return;
}
