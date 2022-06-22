//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sync_eint.cpp
//  \brief Syncs internal with total energy

// Athena++ headers
#include "hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::SyncEint
//  \brief Sync internal energy with total energy, if necessary 

void Hydro::SyncEint(AthenaArray<Real> &u) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real i2 = pmb->peos->GetIeta2(); 
  Real eint, emax;  
  int dim = 1; 

  // Get dimension
  if (pmb->block_size.nx2 > 1) dim = 2;
  if (pmb->block_size.nx3 > 1) dim = 3;  

  // 1D-case
  if ( dim == 1 ) {
// Pretty sure that vectorization does not allow if statements (loop actions must be known in advance)
    for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//#pragma omp simd
    for (int i=is; i<=ie; ++i) {
      eint = u(fluidnum,IEN,ks,js,i) - 0.5*(SQR(u(fluidnum,IM1,ks,js,i)))/u(fluidnum,IDN,ks,js,i);
      emax = u(fluidnum,IEN,ks,js,i); 

      emax = std::max(emax,u(fluidnum,IEN,ks,js,std::max(is,i-1)));
      emax = std::max(emax,u(fluidnum,IEN,ks,js,std::min(ie,i+1)));
			
      if ((eint/emax > i2) && (eint > 0.0)) {
	u(fluidnum,IIE,ks,js,i) = eint; 
      }
    }
    }
  }
  // 2D-case
  else if ( dim == 2 ) {
    for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int j=js; j<=je; ++j) {
// Pretty sure that vectorization does not allow if statements (loop actions must be known in advance)
//#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        eint = u(fluidnum,IEN,ks,j,i) - 0.5*(SQR(u(fluidnum,IM1,ks,j,i)) + SQR(u(fluidnum,IM2,ks,j,i)))/u(fluidnum,IDN,ks,j,i);
        emax = u(fluidnum,IEN,ks,j,i); 

        // this is should be equivalent to the commented out for-loop below
        //emax = std::max(emax,u(IEN,ks,std::max(js,j-1),std::max(is,i-1)));
        //emax = std::max(emax,u(IEN,ks,std::max(js,j-1),            i   ));
        //emax = std::max(emax,u(IEN,ks,std::max(js,j-1),std::min(ie,i+1)));
        //emax = std::max(emax,u(IEN,ks,            j   ,std::max(is,i-1)));
        //emax = std::max(emax,u(IEN,ks,            j   ,            i   ));
        //emax = std::max(emax,u(IEN,ks,            j   ,std::min(ie,i+1)));
        //emax = std::max(emax,u(IEN,ks,std::min(je,j+1),std::max(is,i-1)));
        //emax = std::max(emax,u(IEN,ks,std::min(je,j+1),            i   ));
        //emax = std::max(emax,u(IEN,ks,std::min(je,j+1),std::min(ie,i+1)));
				
        for (int jj=std::max(js,j-1); jj<=std::min(je,j+1); jj++) {
          for (int ii=std::max(is,i-1); ii<=std::min(ie,i+1); ii++) {
            emax = std::max(emax,u(fluidnum,IEN,ks,jj,ii));
          }
        }
        if ((eint/emax > i2) && (eint > 0.0)) {
          //std::cout << "[SyncEint]: k " << ks << " j " << j  << " i "  << i << " eint=" << eint
          //          << " IE: " << u(IIE,ks,j,i) << " ieta2: " << i2 << std::endl;
          u(fluidnum,IIE,ks,j,i) = eint; 
        }
      }
    }
    }
  }
  // 3D-case
  else {
    for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
// Pretty sure that vectorization does not allow if statements (loop actions must be known in advance)
//#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          eint = u(fluidnum,IEN,k,j,i) - 0.5*(SQR(u(fluidnum,IM1,k,j,i)) + SQR(u(fluidnum,IM2,k,j,i)) + SQR(u(fluidnum,IM3,k,j,i)))/u(fluidnum,IDN,k,j,i);
          emax = u(fluidnum,IEN,k,j,i); 

          // this is should be equivalent to the commented out for-loop below
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),std::max(js,j-1),std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),std::max(js,j-1),            i   ));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),std::max(js,j-1),std::min(ie,i+1)));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),            j   ,std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),            j   ,	           i   ));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),            j   ,std::min(ie,i+1)));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),std::min(je,j+1),std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),std::min(je,j+1),            i   ));
          //emax = std::max(emax,u(IEN,std::max(ks,k-1),std::min(je,j+1),std::min(ie,i+1)));

          //emax = std::max(emax,u(IEN,            k   ,std::max(js,j-1),std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,            k   ,std::max(js,j-1),            i   ));
          //emax = std::max(emax,u(IEN,            k   ,std::max(js,j-1),std::min(ie,i+1)));
          //emax = std::max(emax,u(IEN,            k   ,            j   ,std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,            k   ,            j   ,            i   ));
          //emax = std::max(emax,u(IEN,            k   ,            j   ,std::min(ie,i+1)));
          //emax = std::max(emax,u(IEN,            k   ,std::min(je,j+1),std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,            k   ,std::min(je,j+1),            i   ));
          //emax = std::max(emax,u(IEN,            k   ,std::min(je,j+1),std::min(ie,i+1)));

          //emax = std::max(emax,u(IEN,std::min(ke,k-1),std::max(js,j-1),std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),std::max(js,j-1),            i   ));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),std::max(js,j-1),std::min(ie,i+1)));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),            j   ,std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),            j   ,            i   ));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),            j   ,std::min(ie,i+1)));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),std::min(je,j+1),std::max(is,i-1)));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),std::min(je,j+1),            i   ));
          //emax = std::max(emax,u(IEN,std::min(ke,k-1),std::min(je,j+1),std::min(ie,i+1)));

          for (int kk=std::max(ks,k-1); kk<=std::min(ke,k+1); kk++) {
            for (int jj=std::max(js,j-1); jj<=std::min(je,j+1); jj++) {
              for (int ii=std::max(is,i-1); ii<=std::min(ie,i+1); ii++) {
                emax = std::max(emax,u(fluidnum,IEN,kk,jj,ii));
              }
            }
          }
          if ((eint/emax > i2) && (eint > 0.0)) {
            u(fluidnum,IIE,k,j,i) = eint; 
          }
        }
      }
    }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CheckEint
//  \brief Check if internal energy is necessary at start of each substep  

void Hydro::CheckEint(AthenaArray<Real> &u) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real i1 = pmb->peos->GetIeta1(); 
  Real eint, etot, kine;  
  int dim = 1; 

  // Get dimension
  if (pmb->block_size.nx2 > 1) dim = 2;
  if (pmb->block_size.nx3 > 1) dim = 3;  

  // 1D-case
  if ( dim == 1 ) {
    for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//#pragma omp simd
    for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
      etot = u(fluidnum,IEN,ks,js,i);
      kine = 0.5*(SQR(u(fluidnum,IM1,ks,js,i)))/u(fluidnum,IDN,ks,js,i);
      eint = etot - kine;
      if ( ((eint/etot) < i1) || (etot <= 0.0) || (eint <= 0.0) ) {
        u(fluidnum,IEN,ks,js,i)  = u(fluidnum,IIE,ks,js,i) + kine;
      }
    }
    }
  }
  // 2D-case
  else if ( dim == 2 ) {
    for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
//#pragma omp simd
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        etot = u(fluidnum,IEN,ks,j,i);
        kine = 0.5*(SQR(u(fluidnum,IM1,ks,j,i))+SQR(u(fluidnum,IM2,ks,j,i)))/u(fluidnum,IDN,ks,j,i);
        eint = etot - kine;
        if ( ((eint/etot) < i1) || (etot <= 0.0) || (eint <= 0.0) ) {
          u(fluidnum,IEN,ks,j,i)  = u(fluidnum,IIE,ks,j,i) + kine;
        }
      }
    }
    }
  }
  // 3D-case
  else {
    for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int k=ks-NGHOST; k<=ke+NGHOST; ++k) {
      for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
//#pragma omp simd
        for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
          etot = u(fluidnum,IEN,k,j,i); 
          kine = 0.5*(SQR(u(fluidnum,IM1,k,j,i))+SQR(u(fluidnum,IM2,k,j,i))+SQR(u(fluidnum,IM3,k,j,i)) )/u(fluidnum,IDN,k,j,i);
          eint = etot - kine; 
          if ( ((eint/etot) < i1) || (etot <= 0.0) || (eint <= 0.0) ) {
            u(fluidnum,IEN,k,j,i)  = u(fluidnum,IIE,k,j,i) + kine;
          }
        }
      }
    }
    }
  }
  return;
}
