//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file convection.cpp
//  \brief Problem generator for convection
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

Real theat, x1len, x2len, amp;

static void stop_this();
void ConvectionInnerX2(MeshBlock *pmb, Coordinates *pco,
                       AthenaArray<Real> &a,
                       FaceField &b, Real time, Real dt,
                       int is, int ie, int js, int je, int ks, int ke, int ngh);

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {

  EnrollUserBoundaryFunction(INNER_X2, ConvectionInnerX2);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Setup for shell sweep-up. Assumes shell is located at center of box, with
//  coordinates running from -L to L.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  int iprob;
  Real x1min,x1max,x2min,x2max;
  Real gamma = peos->GetGamma();
  Real gm1 = gamma-1.0;
  std::stringstream msg;
#ifdef MPI_PARALLEL
  int mpierr, myid = Globals::my_rank;
#endif

  iprob    = pin->GetInteger("problem","iprob"); // 0: constant density, 1: gaussian perturbation
  theat    = pin->GetOrAddReal("problem","theat",1.0); // heating timescale
  amp      = pin->GetOrAddReal("problem","amp",0.01); // amplitude of temperature perturbation
  x1min    = pin->GetReal("mesh","x1min");
  x1max    = pin->GetReal("mesh","x1max");
  x2min    = pin->GetReal("mesh","x2min");
  x2max    = pin->GetReal("mesh","x2max");
  x1len    = x1max-x1min;
  x2len    = x2max-x2min;


  // Periodic box with constant density and turbulent velocity field
  if (iprob == 0) { 
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        Real x2v = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x1v = pcoord->x1v(i);
          phydro->u(IDN,k,j,i) = std::pow((1.0-(gm1/gamma)*x2v),1.0/gm1)
                                *(1.0-amp*0.5*(1.0+std::cos(2.0*PI*x1v/x1len))*std::exp(-5.0*x2v/x2len));
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
          phydro->u(IEN,k,j,i) =  std::pow((1.0-(gm1/gamma)*pcoord->x2v(j)),gamma/gm1)/gm1 
                                + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                       +SQR(phydro->u(IM2,k,j,i))
                                       +SQR(phydro->u(IM3,k,j,i)))
                                     /phydro->u(IDN,k,j,i);
          //phydro->u(IIE,k,j,i) = std::pow((1.0-gm1*pcoord->x2v(j)),gamma/gm1)/gm1;
        }
      }
    } 
  }

}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {

}


//----------------------------------------------------------------------------------------
//! \fn void ConvectionInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, const Real time, const Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief Set bottom to fixed values.
//----------------------------------------------------------------------------------------
//

void ConvectionInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {

  Real gamma = pmb->peos->GetGamma();

  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVY)) {
      for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(IVY,k,js-j,i) = -prim(IVY,k,js+j-1,i);  // reflect 2-velocity
        }
      }}
    } else {
      if (n==(IDN)) {
        for (int k=ks; k<=ke; ++k) {
          for (int j=1; j<=ngh; ++j) {
            Real x2v = pco->x2v(js-j);
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x1v = pco->x1v(i);
              prim(n,k,js-j,i) = 1.0/(1.0+time/theat)
                                *(1.0-0.5*amp*(1.0+std::cos(2.0*PI*x1v/x1len))*std::exp(-5.0*x2v/x2len));
            }
          }
        }
      } else {
        if (n==(IPR)) {
          for (int k=ks; k<=ke; ++k) {
            for (int j=1; j<=ngh; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                prim(n,k,js-j,i) = std::pow(1.0-((gamma-1)/gamma)*pco->x2v(js-j),gamma/(gamma-1.0));
              }
            }
          }
        } else {
          for (int k=ks; k<=ke; ++k) {
            for (int j=1; j<=ngh; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                prim(n,k,js-j,i) = prim(n,k,js+j-1,i);
              }
            }
          }
        }
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j  ),i) =  b.x1f(k,(js+j-1),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j+1),i) = -b.x2f(k,(js+j-1),i);  // reflect 2-field
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j  ),i) =  b.x3f(k,(js+j-1),i);
      }
    }}
  }

  return;
}


