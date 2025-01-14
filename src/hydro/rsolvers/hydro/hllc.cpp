//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hllc.cpp
//  \brief HLLC Riemann solver for hydrodynamics, an extension of the HLLE fluxes to
//  include the contact wave.  Only works for adiabatic hydrodynamics.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.
//
// - P. Batten, N. Clarke, C. Lambert, and D. M. Causon, "On the Choice of Wavespeeds
//   for the HLLC Riemann Solver", SIAM J. Sci. & Stat. Comp. 18, 6, 1553-1570, (1997).

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief The HLLC Riemann solver for adiabatic hydrodynamics (use HLLE for isothermal)

void Hydro::RiemannSolver(const int kl, const int ku, const int jl, const int ju,
  const int il, const int iu, const int ivx, const AthenaArray<Real> &bx,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr, AthenaArray<Real> &flx,
  AthenaArray<Real> &ey, AthenaArray<Real> &ez) {

  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  Real wli[(NHYDRO)],wri[(NHYDRO)],wroe[(NHYDRO)];
  Real flxi[(NHYDRO)],fl[(NHYDRO)],fr[(NHYDRO)];
  Real gm1 = pmy_block->peos->GetGamma() - 1.0;
  Real igm1 = 1.0/gm1;
  
  Expansion *ex = pmy_block->pex;
  AthenaArray<Real> &eFlx = ex->expFlux[(ivx-1)];;
  AthenaArray<Real> &eVel = ex->vf[(ivx-1)];
  bool move;
  if (EXPANDING_ENABLED) {
    move  = false;
    if ((ivx == IVX)&&(ex->x1Move)){
      move = true;
    } else if ((ivx == IVY)&&(ex->x2Move)) {
      move = true;
    } else if ((ivx == IVZ)&&(ex->x3Move)){
      move = true;
    }
  }

  int n;
  Real wi[(NHYDRO)];
  Real wallV = 0.0;
  Real e;

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma distribute_point
#pragma omp simd private(n,wli,wri,wroe,flxi,fl,fr,wi,wallV,e)
  for (int i=il; i<=iu; ++i) {

//--- Step 1.  Load L/R states into local variables

    wli[IDN]=wl(IDN,k,j,i);
    wli[IVX]=wl(ivx,k,j,i);
    wli[IVY]=wl(ivy,k,j,i);
    wli[IVZ]=wl(ivz,k,j,i);
    wli[IPR]=wl(IPR,k,j,i);
    if (DUAL_ENERGY) 
      wli[IGE]=wl(IGE,k,j,i);
    for (n=(NHYDRO-NSCALARS); n<NHYDRO; n++)
      wli[n] = wl(n,k,j,i); 

    wri[IDN]=wr(IDN,k,j,i);
    wri[IVX]=wr(ivx,k,j,i);
    wri[IVY]=wr(ivy,k,j,i);
    wri[IVZ]=wr(ivz,k,j,i);
    wri[IPR]=wr(IPR,k,j,i);
    if (DUAL_ENERGY) 
      wri[IGE]=wr(IGE,k,j,i);
    for (n=(NHYDRO-NSCALARS); n<NHYDRO; n++)
      wri[n] = wr(n,k,j,i);

//--- Step2.  Compute Roe-averaged state

    Real sqrtdl = std::sqrt(wli[IDN]);
    Real sqrtdr = std::sqrt(wri[IDN]);
    Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

    //    wroe[IDN] = sqrtdl*sqrtdr; // unused in signal velocity estimates
    wroe[IVX] = (sqrtdl*wli[IVX] + sqrtdr*wri[IVX])*isdlpdr;
    wroe[IVY] = (sqrtdl*wli[IVY] + sqrtdr*wri[IVY])*isdlpdr;
    wroe[IVZ] = (sqrtdl*wli[IVZ] + sqrtdr*wri[IVZ])*isdlpdr;

    // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
    // rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
    Real el,er,hroe;
    el = wli[IPR]*igm1 + 0.5*wli[IDN]*(SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ]));
    er = wri[IPR]*igm1 + 0.5*wri[IDN]*(SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ]));
    hroe = ((el + wli[IPR])/sqrtdl + (er + wri[IPR])/sqrtdr)*isdlpdr;

//--- Step 3.  Compute sound speed in L,R, and Roe-averaged states

    Real cl = pmy_block->peos->SoundSpeed(wli);
    Real cr = pmy_block->peos->SoundSpeed(wri);
    Real q = hroe - 0.5*(SQR(wroe[IVX]) + SQR(wroe[IVY]) + SQR(wroe[IVZ]));
    Real a = (q < 0.0) ? 0.0 : std::sqrt(gm1*q);

//--- Step 4.  Compute the max/min wave speeds based on L/R and Roe-averaged values

    Real al = std::min((wroe[IVX] - a),(wli[IVX] - cl));
    Real ar = std::max((wroe[IVX] + a),(wri[IVX] + cr));

    Real bp = ar > 0.0 ? ar : 0.0;
    Real bm = al < 0.0 ? al : 0.0;

//--- Step 5.  Compute the contact wave speed and pressure

    Real vxl = wli[IVX] - al;
    Real vxr = wri[IVX] - ar;

    Real tl = wli[IPR] + vxl*wli[IDN]*wli[IVX];
    Real tr = wri[IPR] + vxr*wri[IDN]*wri[IVX];

    Real ml =   wli[IDN]*vxl;
    Real mr = -(wri[IDN]*vxr);

    // Determine the contact wave speed...
    Real am = (tl - tr)/(ml + mr);
    // ...and the pressure at the contact surface
    Real cp = (ml*tr + mr*tl)/(ml + mr);
    cp = cp > 0.0 ? cp : 0.0;

    // No loop-carried dependencies anywhere in this loop
    //    #pragma distribute_point
//--- Step 6.  Compute L/R fluxes along the line bm, bp

    vxl = wli[IVX] - bm;
    vxr = wri[IVX] - bp;

    fl[IDN] = wli[IDN]*vxl;
    fr[IDN] = wri[IDN]*vxr;

    fl[IVX] = wli[IDN]*wli[IVX]*vxl + wli[IPR];
    fr[IVX] = wri[IDN]*wri[IVX]*vxr + wri[IPR];

    fl[IVY] = wli[IDN]*wli[IVY]*vxl;
    fr[IVY] = wri[IDN]*wri[IVY]*vxr;

    fl[IVZ] = wli[IDN]*wli[IVZ]*vxl;
    fr[IVZ] = wri[IDN]*wri[IVZ]*vxr;

    fl[IEN] = el*vxl + wli[IPR]*wli[IVX];
    fr[IEN] = er*vxr + wri[IPR]*wri[IVX];

//--- Step 8.  Compute flux weights or scales

    Real sl,sr,sm;
    if (am >= 0.0) {
      sl =  am/(am - bm);
      sr = 0.0;
      sm = -bm/(am - bm);
    } else {
      sl =  0.0;
      sr = -am/(bp - am);
      sm =  bp/(bp - am);
    }

//--- Step 9.  Compute the HLLC flux at interface, including the weighted contribution
// of the flux along the contact

    flxi[IDN] = sl*fl[IDN] + sr*fr[IDN];
    flxi[IVX] = sl*fl[IVX] + sr*fr[IVX] + sm*cp;
    flxi[IVY] = sl*fl[IVY] + sr*fr[IVY];
    flxi[IVZ] = sl*fl[IVZ] + sr*fr[IVZ];
    flxi[IEN] = sl*fl[IEN] + sr*fr[IEN] + sm*cp*am;

    flx(IDN,k,j,i) = flxi[IDN];
    flx(ivx,k,j,i) = flxi[IVX];
    flx(ivy,k,j,i) = flxi[IVY];
    flx(ivz,k,j,i) = flxi[IVZ];
    flx(IEN,k,j,i) = flxi[IEN];

    if (DUAL_ENERGY) // IGE is pressure
      flx(IIE,k,j,i) = (flxi[IDN] >= 0 ? flxi[IDN]*wli[IGE]/wli[IDN] : flxi[IDN]*wri[IGE]/wri[IDN])*igm1;

    for (n=(NHYDRO-NSCALARS); n<NHYDRO; n++) 
      flx(n,k,j,i)   = (flxi[IDN] >= 0 ? flxi[IDN]*wli[n] : flxi[IDN]*wri[n]);

    //For Time Dependent grid, account for Wall Flux
    if ((EXPANDING_ENABLED) && (move)) {
      //--- Step 1. Determine Flux Direction
      if (ivx == IVX){
        wallV = eVel(i);
      } else if (ivx == IVY) {
        wallV = eVel(j);
      } else if (ivx == IVZ){
        wallV = eVel(k);
      } else {
        wallV = 0.0;
      }
      //--- Step 2. Load primitive Variables
      if (wallV > 0.0) {
        for (n=0; n<NHYDRO; ++n) 
          wi[n] = wri[n];
      } else if (wallV < 0.0) {
        for (n=0; n<NHYDRO; ++n)
          wi[n] = wli[n];
      } else {
        for (n=0; n<NHYDRO; ++n)
          wi[n] = 0.0;
      }

      e = wi[IPR]*igm1 + 0.5*wi[IDN]*(SQR(wi[IVX]) + SQR(wi[IVY]) + SQR(wi[IVZ]));
      eFlx(IDN,k,j,i) = wi[IDN]*wallV;
      eFlx(ivx,k,j,i) = wi[IDN]*wi[IVX]*wallV;
      eFlx(ivy,k,j,i) = wi[IDN]*wi[IVY]*wallV;
      eFlx(ivz,k,j,i) = wi[IDN]*wi[IVZ]*wallV;
      eFlx(IEN,k,j,i) = e*wallV;
      if (DUAL_ENERGY) 
        eFlx(IIE,k,j,i) = wi[IGE]*wallV*igm1; // IGE is pressure
      for (n=(NHYDRO-NSCALARS); n<NHYDRO; n++) 
        eFlx(n,k,j,i) = wi[IDN]*wi[n]*wallV;
      //fprintf(stdout,"eFlx=%13.5e k,j,i=%3i %3i %3i\n",eFlx(IDN,k,j,i),k,j,i);

    } //End Expanding
  }
  }}

  return;
}
