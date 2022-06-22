//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//  \brief computes timestep using CFL condition on a MEshBlock

// C/C++ headers
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()

// Athena++ headers
#include "hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../cless/cless.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
// \!fn Real Hydro::NewBlockTimeStep(void)
// \brief calculate the minimum timestep within a MeshBlock

Real Hydro::NewBlockTimeStep(void) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> w,bcc,b_x1f,b_x2f,b_x3f, wcl;
  w.InitWithShallowCopy(pmb->phydro->w);
  if (MAGNETIC_FIELDS_ENABLED) {
    bcc.InitWithShallowCopy(pmb->pfield->bcc);
    b_x1f.InitWithShallowCopy(pmb->pfield->b.x1f);
    b_x2f.InitWithShallowCopy(pmb->pfield->b.x2f);
    b_x3f.InitWithShallowCopy(pmb->pfield->b.x3f);
  }
  if (CLESS_ENABLED) {
    wcl.InitWithShallowCopy(pmb->pcless->w); 
  }

  AthenaArray<Real> dt1, dt2, dt3;
  dt1.InitWithShallowCopy(dt1_);
fprintf(stdout,"dt1=%11.3e\n",dt1(2));
  dt2.InitWithShallowCopy(dt2_);
  dt3.InitWithShallowCopy(dt3_);
  Real wi[NFLUIDS][(NWAVE+NINT)];
	Real wicl[(NWAVECL)]; 

  Real min1,min2;

  Real min_dt = (FLT_MAX);
  fprintf(stdout,"FLT_MAX=%13.5e\n",FLT_MAX);

  for (int fluidnum=0; fluidnum<(NFLUIDS); fluidnum++){
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CenterWidth1(k,j,is,ie,dt1);
      pmb->pcoord->CenterWidth2(k,j,is,ie,dt2);
      pmb->pcoord->CenterWidth3(k,j,is,ie,dt3);
      fprintf(stdout,"yeha! dt1=%11.3e\n",dt1(2));
      if (!RELATIVISTIC_DYNAMICS) {
#pragma ivdep
        for (int i=is; i<=ie; ++i) {
          wi[fluidnum][IDN]=w(fluidnum,IDN,k,j,i);
          wi[fluidnum][IVX]=w(fluidnum,IVX,k,j,i);
          wi[fluidnum][IVY]=w(fluidnum,IVY,k,j,i);
          wi[fluidnum][IVZ]=w(fluidnum,IVZ,k,j,i);
          if (NON_BAROTROPIC_EOS) {
            wi[fluidnum][IPR]=w(fluidnum,IPR,k,j,i);
            if (DUAL_ENERGY) wi[fluidnum][IGE]=w(fluidnum,IGE,k,j,i);
          }
					
          if (CLESS_ENABLED) { // cless + hydro
            Real c1f, c2f, c3f; 
            wicl[IDN ] = wcl(IDN ,k,j,i);
            wicl[IVX ] = wcl(IVX ,k,j,i);
            wicl[IVY ] = wcl(IVY ,k,j,i);
            wicl[IVZ ] = wcl(IVZ ,k,j,i);
            wicl[IP11] = wcl(IP11,k,j,i);
            wicl[IP22] = wcl(IP22,k,j,i);
            wicl[IP33] = wcl(IP33,k,j,i);
            wicl[IP12] = wcl(IP12,k,j,i);
            wicl[IP13] = wcl(IP13,k,j,i);
            wicl[IP23] = wcl(IP23,k,j,i);

            pmb->peos->SoundSpeedsCL(wicl,&c1f,&c2f,&c3f); 

            Real temp1[(NWAVE+NINT)],temp2[(NWAVE+NINT)];
            for (int idx = 0; idx < NHYDRO; idx++) {
              temp1[idx] = w(0,idx,k,j,i);
              if(NFLUIDS!=1){
                temp2[idx] = w(1,idx,k,j,i);
              }
            }
// when you're done with it, deallocate
            Real cs = pmb->peos->SoundSpeed(temp1);
            if (RIEMANN_SOLVER=="bgk2n"){
              cs = pmb->peos->SoundSpeed2(temp1,temp2,fluidnum); 
            }
            //delete [] temp;

            dt1(i) /= std::max( fabs(wi[fluidnum][IVX] + cs), fabs(wicl[IVX] + c1f) );
            dt2(i) /= std::max( fabs(wi[fluidnum][IVY] + cs), fabs(wicl[IVY] + c2f) );
            dt3(i) /= std::max( fabs(wi[fluidnum][IVZ] + cs), fabs(wicl[IVZ] + c3f) ); 
          }
          else if (MAGNETIC_FIELDS_ENABLED) { // hydro + mhd

            Real bx = bcc(IB1,k,j,i) + fabs(b_x1f(k,j,i)-bcc(IB1,k,j,i));
            wi[fluidnum][IBY] = bcc(IB2,k,j,i);
            wi[fluidnum][IBZ] = bcc(IB3,k,j,i);

            Real tempb1[(NWAVE+NINT)];
            for (int idx = 0; idx < NHYDRO; idx++) {
              tempb1[idx] = wi[fluidnum][idx];
            }

            Real cf = pmb->peos->FastMagnetosonicSpeed(tempb1,bx);
            //delete [] temp;

            dt1(i) /= (fabs(wi[fluidnum][IVX]) + cf);

            wi[fluidnum][IBY] = bcc(IB3,k,j,i);
            wi[fluidnum][IBZ] = bcc(IB1,k,j,i);
            bx = bcc(IB2,k,j,i) + fabs(b_x2f(k,j,i)-bcc(IB2,k,j,i));

            Real tempb2[(NWAVE+NINT)];
            for (int idx = 0; idx < NHYDRO; idx++) {
              tempb2[idx] = wi[fluidnum][idx];
            }

            cf = pmb->peos->FastMagnetosonicSpeed(tempb2,bx);
            //delete [] temp2;

            dt2(i) /= (fabs(wi[fluidnum][IVY]) + cf);

            wi[fluidnum][IBY] = bcc(IB1,k,j,i);
            wi[fluidnum][IBZ] = bcc(IB2,k,j,i);
            bx = bcc(IB3,k,j,i) + fabs(b_x3f(k,j,i)-bcc(IB3,k,j,i));

            Real tempb3[(NWAVE+NINT)];
            for (int idx = 0; idx < NHYDRO; idx++) {
              tempb3[idx] = wi[fluidnum][idx];
            }

            cf = pmb->peos->FastMagnetosonicSpeed(tempb3,bx);
            //delete [] temp3;

            dt3(i) /= (fabs(wi[fluidnum][IVZ]) + cf);

          } 
          else { // hydro only 

            Real temp1[(NWAVE+NINT)],temp2[(NWAVE+NINT)];
            for (int idx = 0; idx < NHYDRO; idx++) {
              temp1[idx] = w(0,idx,k,j,i);
              temp2[idx] = w(1,idx,k,j,i);
            }

            Real cs = pmb->peos->SoundSpeed(temp1);
            if (RIEMANN_SOLVER=="bgk2n"){
              cs = pmb->peos->SoundSpeed2(temp1,temp2,fluidnum);
            }
 //delete [] temp;
      
           fprintf(stdout,"dt1=%11.3e,wi=%11.3e\n",dt1(i),wi[fluidnum][IVX]);

            dt1(i) /= (fabs(wi[fluidnum][IVX]) + cs);
            dt2(i) /= (fabs(wi[fluidnum][IVY]) + cs);
            dt3(i) /= (fabs(wi[fluidnum][IVZ]) + cs);

            fprintf(stdout,"temp1[1]=%13.5e,temp2[1]=%13.5e,fluidnum=%4i\n",temp1[1],temp2[1],fluidnum);
            fprintf(stdout,"wi=%13.5e, i=%4i, j=%4i, dt1(i)=%13.5e, cs=%13.5e\n",wi[fluidnum][IVX],i,j,dt1(i),cs);


          }
        }
      }

      // compute minimum of (v1 +/- C)
      for (int i=is; i<=ie; ++i) {
        Real& dt_1 = dt1(i);
        fprintf(stdout,"fluidnum=%4i,i=%4i,j=%4i,min_dt=%13.5e,dt1(i)=%13.5e\n",fluidnum,i,j,min_dt,dt1(i));
        min_dt = std::min(min_dt,dt_1);
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
        fprintf(stdout,"dim check\n");
        for (int i=is; i<=ie; ++i) {
          Real& dt_2 = dt2(i);
          min_dt = std::min(min_dt,dt_2);
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_3 = dt3(i);
          min_dt = std::min(min_dt,dt_3);
        }
      }
      
      if(fluidnum==0){
        min1=min_dt;
      }else if(fluidnum==1){  
        min2=min_dt;
      }
    }
  }}

// calculate the timestep limited by the diffusion process
  if (phdif->hydro_diffusion_defined) {
    Real mindt_vis, mindt_cnd;
    phdif->NewHydroDiffusionDt(mindt_vis, mindt_cnd);
    min_dt = std::min(min_dt,mindt_vis);
    min_dt = std::min(min_dt,mindt_cnd);
    fprintf(stdout,"DIFFUSION, mindt_vis=%11.3e, mindt_cnd=%11.3e\n",mindt_vis,mindt_cnd);
  } // hydro diffusion

  if(MAGNETIC_FIELDS_ENABLED &&
     pmb->pfield->pfdif->field_diffusion_defined) {
    Real mindt_oa, mindt_h;
    pmb->pfield->pfdif->NewFieldDiffusionDt(mindt_oa, mindt_h);
    min_dt = std::min(min_dt,mindt_oa);
    min_dt = std::min(min_dt,mindt_h);
  } // field diffusion

  if(NFLUIDS!=1){
    min_dt=std::min(min1,min2);
  }

  fprintf(stdout,"min_dt=%11.5e\n",min_dt);
  min_dt *= pmb->pmy_mesh->cfl_number;

  if (UserTimeStep_!=NULL) {
    min_dt = std::min(min_dt, UserTimeStep_(pmb));
  }


  pmb->new_block_dt=min_dt;
  fprintf(stdout,"min_dt=%11.5e\n",min_dt);
  return min_dt;
}
