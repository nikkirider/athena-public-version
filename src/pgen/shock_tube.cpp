//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shock_tube.cpp
//  \brief Problem generator for shock tube problems.
//
// Problem generator for shock tube (1-D Riemann) problems. Initializes plane-parallel
// shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D), and along x3 (in 3D).
//========================================================================================

// C headers
#include <stdio.h>

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../cless/cless.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Calculate L1 errors in Sod (hydro) and RJ2a (MHD) tests
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  MeshBlock *pmb = pblock;

  if (!pin->GetOrAddBoolean("problem","compute_error",false)) return;

  // Read shock direction and set array indices
  int shk_dir = pin->GetInteger("problem","shock_dir");
  int im1,im2,im3,ib1,ib2,ib3;
  if (shk_dir == 1) {
    im1 = IM1; im2 = IM2; im3 = IM3;
    ib1 = IB1; ib2 = IB2; ib3 = IB3;
  } else if (shk_dir == 2) {
    im1 = IM2; im2 = IM3; im3 = IM1;
    ib1 = IB2; ib2 = IB3; ib3 = IB1;
  } else {
    im1 = IM3; im2 = IM1; im3 = IM2;
    ib1 = IB3; ib2 = IB1; ib3 = IB2;
  }

  // Initialize errors to zero
  Real err[NHYDRO+NFIELD];
  for (int i=0; i<(NHYDRO+NFIELD); ++i) err[i]=0.0;

  // Errors in RJ2a test (Dai & Woodward 1994 Tables Ia and Ib)
  if (MAGNETIC_FIELDS_ENABLED) {
    Real xfp = 2.2638*tlim;
    Real xrp = (0.53432 + 1.0/std::sqrt(PI*1.309))*tlim;
    Real xsp = (0.53432 + 0.48144/1.309)*tlim;
    Real xc = 0.57538*tlim;
    Real xsm = (0.60588 - 0.51594/1.4903)*tlim;
    Real xrm = (0.60588 - 1.0/std::sqrt(PI*1.4903))*tlim;
    Real xfm = (1.2 - 2.3305/1.08)*tlim;
   
    for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    Real gm1 = pmb->peos->GetGamma() - 1.0;
    if(fluidnum==1){
      Real gm1  = pmb->peos->GetGamma2() - 1.0;
    }
    for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        Real r, d0, mx, my, mz, e0, bx, by, bz;
        if (shk_dir == 1) r = pmb->pcoord->x1v(i);
        if (shk_dir == 2) r = pmb->pcoord->x2v(j);
        if (shk_dir == 3) r = pmb->pcoord->x3v(k);

        bx = 2.0/std::sqrt(4.0*PI);
        if (r > xfp) {
          d0 = 1.0;
          mx = 0.0;
          my = 0.0;
          mz = 0.0;
          by = 4.0/std::sqrt(4.0*PI);
          bz = 2.0/std::sqrt(4.0*PI);
          e0 = 1.0/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else if (r > xrp) {
          d0 = 1.3090;
          mx = 0.53432*d0;
          my = -0.094572*d0;
          mz = -0.047286*d0;
          by = 5.3452/std::sqrt(4.0*PI);
          bz = 2.6726/std::sqrt(4.0*PI);
          e0 = 1.5844/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else if (r > xsp) {
          d0 = 1.3090;
          mx = 0.53432*d0;
          my = -0.18411*d0;
          mz = 0.17554*d0;
          by = 5.7083/std::sqrt(4.0*PI);
          bz = 1.7689/std::sqrt(4.0*PI);
          e0 = 1.5844/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else if (r > xc) {
          d0 = 1.4735;
          mx = 0.57538*d0;
          my = 0.047601*d0;
          mz = 0.24734*d0;
          by = 5.0074/std::sqrt(4.0*PI);
          bz = 1.5517/std::sqrt(4.0*PI);
          e0 = 1.9317/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else if (r > xsm) {
          d0 = 1.6343;
          mx = 0.57538*d0;
          my = 0.047601*d0;
          mz = 0.24734*d0;
          by = 5.0074/std::sqrt(4.0*PI);
          bz = 1.5517/std::sqrt(4.0*PI);
          e0 = 1.9317/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else if (r > xrm) {
          d0 = 1.4903;
          mx = 0.60588*d0;
          my = 0.22157*d0;
          mz = 0.30125*d0;
          by = 5.5713/std::sqrt(4.0*PI);
          bz = 1.7264/std::sqrt(4.0*PI);
          e0 = 1.6558/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else if (r > xfm) {
          d0 = 1.4903;
          mx = 0.60588*d0;
          my = 0.11235*d0;
          mz = 0.55686*d0;
          by = 5.0987/std::sqrt(4.0*PI);
          bz = 2.8326/std::sqrt(4.0*PI);
          e0 = 1.6558/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        } else {
          d0 = 1.08;
          mx = 1.2*d0;
          my = 0.01*d0;
          mz = 0.5*d0;
          by = 3.6/std::sqrt(4.0*PI);
          bz = 2.0/std::sqrt(4.0*PI);
          e0 = 0.95/gm1 + 0.5*((mx*mx+my*my+mz*mz)/d0 + (bx*bx+by*by+bz*bz));
        }

        err[IDN] += fabs(d0 - pmb->phydro->u(fluidnum,IDN,k,j,i));
        err[im1] += fabs(mx - pmb->phydro->u(fluidnum,im1,k,j,i));
        err[im2] += fabs(my - pmb->phydro->u(fluidnum,im2,k,j,i));
        err[im3] += fabs(mz - pmb->phydro->u(fluidnum,im3,k,j,i));
        err[IEN] += fabs(e0 - pmb->phydro->u(fluidnum,IEN,k,j,i));
        err[NHYDRO + ib1] += fabs(bx - pmb->pfield->bcc(ib1,k,j,i));
        err[NHYDRO + ib2] += fabs(by - pmb->pfield->bcc(ib2,k,j,i));
        err[NHYDRO + ib3] += fabs(bz - pmb->pfield->bcc(ib3,k,j,i));
      }
    }}}

  // Errors in Sod solution
  } else {
    // Positions of shock, contact, head and foot of rarefaction for Sod test
    Real xs = 1.7522*tlim;
    Real xc = 0.92745*tlim;
    Real xf = -0.07027*tlim;
    Real xh = -1.1832*tlim;

    for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
    for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        Real r,d0,m0,e0;
        if (shk_dir == 1) r = pmb->pcoord->x1v(i);
        if (shk_dir == 2) r = pmb->pcoord->x2v(j);
        if (shk_dir == 3) r = pmb->pcoord->x3v(k);

        if (r > xs) {
          d0 = 0.125;
          m0 = 0.0;
          e0 = 0.25;
        } else if (r > xc) {
          d0 = 0.26557;
          m0 = 0.92745*d0;
          e0 = 0.87204;
        } else if (r > xf) {
          d0 = 0.42632;
          m0 = 0.92745*d0;
          e0 = 0.94118;
        } else if (r > xh) {
          Real v0 = 0.92745*(r-xh)/(xf-xh);
          d0 = 0.42632*pow((1.0+0.20046*(0.92745-v0)),5);
          m0 = v0*d0;
          e0 = (0.30313*pow((1.0+0.20046*(0.92745-v0)),7))/0.4 + 0.5*d0*v0*v0;
        } else {
          d0 = 1.0;
          m0 = 0.0;
          e0 = 2.5;
        }
        err[IDN] += fabs(d0  - pmb->phydro->u(fluidnum,IDN,k,j,i));
        err[im1] += fabs(m0  - pmb->phydro->u(fluidnum,im1,k,j,i));
        err[im2] += fabs(0.0 - pmb->phydro->u(fluidnum,im2,k,j,i));
        err[im3] += fabs(0.0 - pmb->phydro->u(fluidnum,im3,k,j,i));
        err[IEN] += fabs(e0  - pmb->phydro->u(fluidnum,IEN,k,j,i));
      }
    }}}
  }

  // normalize errors by number of cells, compute RMS
  for (int i=0; i<(NHYDRO+NFIELD); ++i) {
    err[i] = err[i]/static_cast<Real>(GetTotalCells());
  }
  Real rms_err = 0.0;
  for (int i=0; i<(NHYDRO+NFIELD); ++i) rms_err += SQR(err[i]);
  rms_err = std::sqrt(rms_err);

  // open output file and write out errors
  std::string fname;
  fname.assign("shock-errors.dat");
  std::stringstream msg;
  FILE *pfile;

  // The file exists -- reopen the file in append mode
  if ((pfile = fopen(fname.c_str(),"r")) != NULL) {
    if ((pfile = freopen(fname.c_str(),"a",pfile)) == NULL) {
      msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
          << std::endl << "Error output file could not be opened" <<std::endl;
      throw std::runtime_error(msg.str().c_str());
    }

  // The file does not exist -- open the file in write mode and add headers
  } else {
    if ((pfile = fopen(fname.c_str(),"w")) == NULL) {
      msg << "### FATAL ERROR in function [Mesh::UserWorkAfterLoop]"
          << std::endl << "Error output file could not be opened" <<std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    fprintf(pfile,"# Nx1  Nx2  Nx3  Ncycle  RMS-Error  d  M1  M2  M3  E");
    if (MAGNETIC_FIELDS_ENABLED) fprintf(pfile,"  B1c  B2c  B3c");
    fprintf(pfile,"\n");
  }

  // write errors
  fprintf(pfile,"%d  %d",pmb->block_size.nx1,pmb->block_size.nx2);
  fprintf(pfile,"  %d  %d  %e",pmb->block_size.nx3,ncycle,rms_err);
  fprintf(pfile,"  %e  %e  %e  %e  %e",err[IDN],err[IM1],err[IM2],err[IM3],err[IEN]);
  if (MAGNETIC_FIELDS_ENABLED) {
    fprintf(pfile,"  %e  %e  %e",err[NHYDRO+IB1],err[NHYDRO+IB2],err[NHYDRO+IB3]);
  }
  fprintf(pfile,"\n");
  fclose(pfile);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the shock tube tests
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

//This was specifically to test InitWithShallowSlice2Index:    
//  AthenaArray<int> atest;
//  AthenaArray<int> atest0, atest1;
//  atest.NewAthenaArray(2,10);
//  atest0.NewAthenaArray(10);
//  atest1.NewAthenaArray(10);

//  for(int fluidnum=0;fluidnum<NFLUIDS;fluidnum++){
//    for(int i=0;i<10;i++){
//      for(int j=0;j<10;j++){
//        atest(fluidnum,i)=(fluidnum+1)*10+i;
//      }
//    }
//  }

//  fprintf(stdout,"atest(f1,2)=%4i,atest(f2,2)=%4i\n",atest(0,2),atest(1,2));

//  InitWithShallowSlice2Index(atest,dim,wanted_indx,nvar,fluidnum)
//  atest0.InitWithShallowSlice2Index(atest,2,0,1,0);
//  atest1.InitWithShallowSlice2Index(atest,2,0,1,1);
  
//  for(int i=0;i<10;i++){
//    fprintf(stdout,"atest=%4i%4i atest0=%4i atest1=%4i\n",atest(0,i),atest(1,i),atest0(i),atest1(i));
//  } 
 
//  throw std::exception(); 


  // parse shock direction: {1,2,3} -> {x1,x2,x3}
  int shk_dir = pin->GetInteger("problem","shock_dir");

  // parse shock location (must be inside grid)
  Real xshock = pin->GetReal("problem","xshock");
  if (shk_dir == 1 && (xshock < pmy_mesh->mesh_size.x1min ||
                       xshock > pmy_mesh->mesh_size.x1max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x1 domain for shkdir=" << shk_dir << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (shk_dir == 2 && (xshock < pmy_mesh->mesh_size.x2min ||
                       xshock > pmy_mesh->mesh_size.x2max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x2 domain for shkdir=" << shk_dir << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (shk_dir == 3 && (xshock < pmy_mesh->mesh_size.x3min ||
                       xshock > pmy_mesh->mesh_size.x3max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x3 domain for shkdir=" << shk_dir << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

    AthenaArray<Real> wl0;
    AthenaArray<Real> wr0;

  // Parse left state read from input file: dl,ul,vl,wl,[pl]
    wl0.NewAthenaArray(NFLUIDS,NHYDRO+NFIELD);
    wr0.NewAthenaArray(NFLUIDS,NHYDRO+NFIELD);

    //Real wl[NFLUIDS][NHYDRO+NFIELD];
    //Real wr[NFLUIDS][NHYDRO+NFIELD];

    wl0(0,IDN) = pin->GetReal("problem","dl1");
    wl0(0,IVX) = pin->GetReal("problem","ul1");
    wl0(0,IVY) = pin->GetReal("problem","vl1");
    wl0(0,IVZ) = pin->GetReal("problem","wl1");
    if (NON_BAROTROPIC_EOS) wl0(0,IPR) = pin->GetReal("problem","pl1");

    if(NFLUIDS==2){
      wl0(1,IDN) = pin->GetReal("problem","dl2");
      wl0(1,IVX) = pin->GetReal("problem","ul2");
      wl0(1,IVY) = pin->GetReal("problem","vl2");
      wl0(1,IVZ) = pin->GetReal("problem","wl2");
      if (NON_BAROTROPIC_EOS) wl0(1,IPR) = pin->GetReal("problem","pl2");
    }

    wr0(0,IDN) = pin->GetReal("problem","dr1");
    wr0(0,IVX) = pin->GetReal("problem","ur1");
    wr0(0,IVY) = pin->GetReal("problem","vr1");
    wr0(0,IVZ) = pin->GetReal("problem","wr1");
    if (NON_BAROTROPIC_EOS) wr0(0,IPR) = pin->GetReal("problem","pr1");

    if(NFLUIDS==2){
      wr0(1,IDN) = pin->GetReal("problem","dr2");
      wr0(1,IVX) = pin->GetReal("problem","ur2");
      wr0(1,IVY) = pin->GetReal("problem","vr2");
      wr0(1,IVZ) = pin->GetReal("problem","wr2");
      if (NON_BAROTROPIC_EOS) wr0(1,IPR) = pin->GetReal("problem","pr2");
    }

    Real ramp=pin->GetReal("problem","ramp"); 


// Initialize the discontinuity in the Hydro variables ---------------------------------

  switch(shk_dir) {

//--- shock in 1-direction
  case 1:
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (pcoord->x1v(i) < xshock) {
           for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){

             if(ramp!=0.0){
               if(fluidnum==0){
//                 wl0(fluidnum,IPR) = 0.5*(1.0-tanh((pcoord->x1v(i))/ramp));
                 wl0(fluidnum,IDN) = 0.5*(1.0-tanh((pcoord->x1v(i))/ramp));
               }else if(fluidnum==1){
//                 wl0(fluidnum,IPR) = 0.5*(1.0+tanh((pcoord->x1v(i))/ramp));
                 wl0(fluidnum,IDN) = 0.5*0.125*(1.0+tanh((pcoord->x1v(i))/ramp));
               }
             }

             phydro->u(fluidnum,IDN,k,j,i) = wl0(fluidnum,IDN);
             phydro->u(fluidnum,IM1,k,j,i) = wl0(fluidnum,IVX)*wl0(fluidnum,IDN);
             phydro->u(fluidnum,IM2,k,j,i) = wl0(fluidnum,IVY)*wl0(fluidnum,IDN);
             phydro->u(fluidnum,IM3,k,j,i) = wl0(fluidnum,IVZ)*wl0(fluidnum,IDN);
          


             if (NON_BAROTROPIC_EOS){
                Real gam=peos->GetGamma();
                if(fluidnum==1){
                  gam=peos->GetGamma2();
                }
                phydro->u(fluidnum,IEN,k,j,i) = wl0(fluidnum,IPR)/(gam - 1.0)
                + 0.5*wl0(fluidnum,IDN)*(wl0(fluidnum,IVX)*wl0(fluidnum,IVX) + wl0(fluidnum,IVY)*wl0(fluidnum,IVY)
                + wl0(fluidnum,IVZ)*wl0(fluidnum,IVZ));
             }


      
	     if (CLESS_ENABLED) {
                pcless->u(IDN ,k,j,i) = phydro->u(IDN,k,j,i); 
                pcless->u(IM1 ,k,j,i) = phydro->u(IM1,k,j,i);
                pcless->u(IM2 ,k,j,i) = phydro->u(IM2,k,j,i);
                pcless->u(IM3 ,k,j,i) = phydro->u(IM3,k,j,i);
                pcless->u(IE11,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVX)*wl0(fluidnum,IVX);
                pcless->u(IE22,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVY)*wl0(fluidnum,IVY);
                pcless->u(IE33,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVZ)*wl0(fluidnum,IVZ);
                pcless->u(IE12,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVX)*wl0(fluidnum,IVY);
                pcless->u(IE13,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVX)*wl0(fluidnum,IVZ);
                pcless->u(IE23,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVY)*wl0(fluidnum,IVZ);
             }

          }

      
        }else{
            for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){

             if(ramp!=0.0){
               if(fluidnum==0){
//                 wr0(fluidnum,IPR) = 0.5*(1.0-tanh((pcoord->x1v(i))/ramp));
                 wr0(fluidnum,IDN) = 0.5*(1.0-tanh((pcoord->x1v(i))/ramp));
               }else if(fluidnum==1){
//                 wr0(fluidnum,IPR) = 0.5*(1.0+tanh((pcoord->x1v(i))/ramp));
                 wr0(fluidnum,IDN) = 0.5*0.125*(1.0+tanh((pcoord->x1v(i))/ramp));
               }
             }

              phydro->u(fluidnum,IDN,k,j,i) = wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM1,k,j,i) = wr0(fluidnum,IVX)*wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM2,k,j,i) = wr0(fluidnum,IVY)*wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM3,k,j,i) = wr0(fluidnum,IVZ)*wr0(fluidnum,IDN);
           
 
              if (NON_BAROTROPIC_EOS){
                Real gam=peos->GetGamma();
                if(fluidnum==1){
                  gam=peos->GetGamma2();
               }
               phydro->u(fluidnum,IEN,k,j,i) = wr0(fluidnum,IPR)/(gam - 1.0)
               + 0.5*wr0(fluidnum,IDN)*(wr0(fluidnum,IVX)*wr0(fluidnum,IVX) + wr0(fluidnum,IVY)*wr0(fluidnum,IVY)
               + wr0(fluidnum,IVZ)*wr0(fluidnum,IVZ));
	   
               }


				
               if (CLESS_ENABLED) {
                 pcless->u(IDN ,k,j,i) = phydro->u(IDN,k,j,i); 
                 pcless->u(IM1 ,k,j,i) = phydro->u(IM1,k,j,i);
                 pcless->u(IM2 ,k,j,i) = phydro->u(IM2,k,j,i);
                 pcless->u(IM3 ,k,j,i) = phydro->u(IM3,k,j,i);
                 pcless->u(IE11,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVX)*wr0(fluidnum,IVX);
                 pcless->u(IE22,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVY)*wr0(fluidnum,IVY);
                 pcless->u(IE33,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVZ)*wr0(fluidnum,IVZ);
                 pcless->u(IE12,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVX)*wr0(fluidnum,IVY);
                 pcless->u(IE13,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVX)*wr0(fluidnum,IVZ);
                 pcless->u(IE23,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVY)*wr0(fluidnum,IVZ);
               }
            }
        }
     }}}
    break;

//--- shock in 2-direction
  case 2:
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      if (pcoord->x2v(j) < xshock) {
        for (int i=is; i<=ie; ++i) {
            for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
              phydro->u(fluidnum,IDN,k,j,i) = wl0(fluidnum,IDN);
              phydro->u(fluidnum,IM1,k,j,i) = wl0(fluidnum,IVX)*wl0(fluidnum,IDN);
              phydro->u(fluidnum,IM2,k,j,i) = wl0(fluidnum,IVY)*wl0(fluidnum,IDN);
              phydro->u(fluidnum,IM3,k,j,i) = wl0(fluidnum,IVZ)*wl0(fluidnum,IDN);
    
              if (NON_BAROTROPIC_EOS){
                Real gam=peos->GetGamma();
                if(fluidnum==1){
                  gam=peos->GetGamma2();
                }
                phydro->u(fluidnum,IEN,k,j,i) = wl0(fluidnum,IPR)/(gam - 1.0)
                + 0.5*wl0(fluidnum,IDN)*(wl0(fluidnum,IVX)*wl0(fluidnum,IVX) + wl0(fluidnum,IVY)*wl0(fluidnum,IVY)
                + wl0(fluidnum,IVZ)*wl0(fluidnum,IVZ));
             }
             if (CLESS_ENABLED) {
               pcless->u(IDN ,k,j,i) = phydro->u(IDN,k,j,i); 
               pcless->u(IM2 ,k,j,i) = phydro->u(IM2,k,j,i);
               pcless->u(IM3 ,k,j,i) = phydro->u(IM3,k,j,i);
               pcless->u(IM1 ,k,j,i) = phydro->u(IM1,k,j,i);
               pcless->u(IE22,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVX)*wl0(fluidnum,IVX);
               pcless->u(IE33,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVY)*wl0(fluidnum,IVY);
               pcless->u(IE11,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVZ)*wl0(fluidnum,IVZ);
               pcless->u(IE23,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVX)*wl0(fluidnum,IVY);
               pcless->u(IE12,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVX)*wl0(fluidnum,IVZ);
               pcless->u(IE13,k,j,i) = wl0(fluidnum,IPR) + wl0(fluidnum,IDN)*wl0(fluidnum,IVY)*wl0(fluidnum,IVZ);
             }
          }
       }

      } else {
        for (int i=is; i<=ie; ++i) {
            for(int fluidnum=0;fluidnum<2;fluidnum++){
              phydro->u(fluidnum,IDN,k,j,i) = wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM1,k,j,i) = wr0(fluidnum,IVX)*wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM2,k,j,i) = wr0(fluidnum,IVY)*wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM3,k,j,i) = wr0(fluidnum,IVZ)*wr0(fluidnum,IDN);
            
              if (NON_BAROTROPIC_EOS){
                Real gam=peos->GetGamma();
                if(fluidnum==1){
                  gam=peos->GetGamma2();
                }
                phydro->u(fluidnum,IEN,k,j,i) = wr0(fluidnum,IPR)/(gam - 1.0)
                + 0.5*wr0(fluidnum,IDN)*(wr0(fluidnum,IVX)*wr0(fluidnum,IVX) + wr0(fluidnum,IVY)*wr0(fluidnum,IVY)
                + wr0(fluidnum,IVZ)*wr0(fluidnum,IVZ));
              
               }
               if (CLESS_ENABLED) {
                 pcless->u(IDN ,k,j,i) = phydro->u(IDN,k,j,i); 
                 pcless->u(IM2 ,k,j,i) = phydro->u(IM2,k,j,i);
                 pcless->u(IM3 ,k,j,i) = phydro->u(IM3,k,j,i);
                 pcless->u(IM1 ,k,j,i) = phydro->u(IM1,k,j,i);
                 pcless->u(IE22,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVX)*wr0(fluidnum,IVX);
                 pcless->u(IE33,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVY)*wr0(fluidnum,IVY);
                 pcless->u(IE11,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVZ)*wr0(fluidnum,IVZ);
                 pcless->u(IE23,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVX)*wr0(fluidnum,IVY);
                 pcless->u(IE12,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVX)*wr0(fluidnum,IVZ);
                 pcless->u(IE13,k,j,i) = wr0(fluidnum,IPR) + wr0(fluidnum,IDN)*wr0(fluidnum,IVY)*wr0(fluidnum,IVZ);
               }
            }
         }}
    }}
    break;

//--- shock in 3-direction

  case 3:
    for (int k=ks; k<=ke; ++k) {
      if (pcoord->x3v(k) < xshock) {
        for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
            for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
              phydro->u(fluidnum,IDN,k,j,i) = wl0(fluidnum,IDN);
              phydro->u(fluidnum,IM3,k,j,i) = wl0(fluidnum,IVX)*wl0(fluidnum,IDN);
              phydro->u(fluidnum,IM1,k,j,i) = wl0(fluidnum,IVY)*wl0(fluidnum,IDN);
              phydro->u(fluidnum,IM2,k,j,i) = wl0(fluidnum,IVZ)*wl0(fluidnum,IDN);
            
              if (NON_BAROTROPIC_EOS){
                Real gam=peos->GetGamma();
                if(fluidnum==1){
                  gam=peos->GetGamma2();
                }
                phydro->u(fluidnum,IEN,k,j,i) = wl0(fluidnum,IPR)/(gam - 1.0)
                + 0.5*wl0(fluidnum,IDN)*(wl0(fluidnum,IVX)*wl0(fluidnum,IVX) + wl0(fluidnum,IVY)*wl0(fluidnum,IVY)
                + wl0(fluidnum,IVZ)*wl0(fluidnum,IVZ));
              }
            
            }
          }}
      } else {
          for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
              phydro->u(fluidnum,IDN,k,j,i) = wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM3,k,j,i) = wr0(fluidnum,IVX)*wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM1,k,j,i) = wr0(fluidnum,IVY)*wr0(fluidnum,IDN);
              phydro->u(fluidnum,IM2,k,j,i) = wr0(fluidnum,IVZ)*wr0(fluidnum,IDN);
     
              if (NON_BAROTROPIC_EOS){
                Real gam=peos->GetGamma();
                if(fluidnum==1){
                  gam=peos->GetGamma2();
                }
                phydro->u(fluidnum,IEN,k,j,i) = wr0(fluidnum,IPR)/(gam - 1.0)
                + 0.5*wr0(fluidnum,IDN)*(wr0(fluidnum,IVX)*wr0(fluidnum,IVX) + wr0(fluidnum,IVY)*wr0(fluidnum,IVY)
                + wr0(fluidnum,IVZ)*wr0(fluidnum,IVZ));
           
              }
            }}}
      }
    }
    break;

  default:
    msg << "### FATAL ERROR in Problem Generator" << std::endl
        << "shock_dir=" << shk_dir << " must be either 1,2, or 3" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  wl0.DeleteAthenaArray(); 
  wr0.DeleteAthenaArray();

// now set face-centered (interface) magnetic fields -----------------------------------
//TAKE FROM OG 1-FLUID VERSION IF NEEDED
  
  return;
}
