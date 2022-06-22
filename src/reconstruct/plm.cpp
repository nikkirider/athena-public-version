//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file plm-uniform.cpp
//  \brief  piecewise linear reconstruction for both uniform and non-uniform meshes

//C++ headers
#include <algorithm>
#include <cmath>

// Athena++ headers
#include "reconstruction.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"

#define DEBUG_ALL

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX1()
//  \brief

void Reconstruction::PiecewiseLinearX1(MeshBlock *pmb,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  Coordinates *pco = pmb->pcoord;
  Field *pfield;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> bx,dw2,wc,dwl,dwr,dwm;
  AthenaArray<Real> temp_u,ul,ur,du2,uc,dul,dur,dum;
 
  temp_u.InitWithShallowCopy(pmb->precon->scr15_ni_);
  ul.InitWithShallowCopy(pmb->precon->scr16_ni_);
  ur.InitWithShallowCopy(pmb->precon->scr17_ni_);
  du2.InitWithShallowCopy(pmb->precon->scr02_i_);
  uc.InitWithShallowCopy(pmb->precon->scr1_ni_);
  dul.InitWithShallowCopy(pmb->precon->scr2_ni_);
  dur.InitWithShallowCopy(pmb->precon->scr3_ni_);
  dum.InitWithShallowCopy(pmb->precon->scr4_ni_);
  bx.InitWithShallowCopy(pmb->precon->scr01_i_);
  dw2.InitWithShallowCopy(pmb->precon->scr02_i_);
  wc.InitWithShallowCopy(pmb->precon->scr1_ni_);
  dwl.InitWithShallowCopy(pmb->precon->scr2_ni_);
  dwr.InitWithShallowCopy(pmb->precon->scr3_ni_);
  dwm.InitWithShallowCopy(pmb->precon->scr4_ni_); 

  Real truemin, min1, trigger, iden, sign1, sign2, sign3, sign4;

  if (pmb->precon->cons_reconstruction){

    pmb->peos->PrimitiveToConserved(w,pfield->bcc,temp_u,pco,il-1,iu,jl,ju,kl,ku);

//    fprintf(stdout,"please w=%11.3e, u=%11.3e\n",w(0,IDN,0,28,32),temp_u(0,IDN,0,28,32));

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {

//        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//#pragma omp simd simdlen(SIMD_WIDTH)
//          for (int i=il-1; i<=iu; ++i) {
//            iden=1.0/temp_u(fluidnum,0,k,j,i);
//            temp_u(fluidnum,1,k,j,i)=temp_u(fluidnum,1,k,j,i)*iden;
//            temp_u(fluidnum,2,k,j,i)=temp_u(fluidnum,2,k,j,i)*iden;
//            temp_u(fluidnum,3,k,j,i)=temp_u(fluidnum,3,k,j,i)*iden;
//          }
//        }

    // compute L/R slopes for each variable
        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NHYDRO); ++n) {
#pragma omp simd
            for (int i=il-1; i<=iu; ++i) {
              dul(fluidnum,n,i) = (temp_u(fluidnum,n,k,j,i  ) - temp_u(fluidnum,n,k,j,i-1));
              dur(fluidnum,n,i) = (temp_u(fluidnum,n,k,j,i+1) - temp_u(fluidnum,n,k,j,i  ));
              uc(fluidnum,n,i) = temp_u(fluidnum,n,k,j,i);

              if(i==32||i==33||i==34||i==35||i==36){
                fprintf(stdout,"w=%11.7e,w+1=%11.7e,w-1=%11.7e,n=%4i,i=%4i\n",w(fluidnum,n,k,j,i),w(fluidnum,n,k,j,i+1),w(fluidnum,n,k,j,i-1),n,i);
                fprintf(stdout,"u=%11.7e,u+1=%11.7e,u-1=%11.7e,n=%4i,i=%4i\n",temp_u(fluidnum,n,k,j,i),temp_u(fluidnum,n,k,j,i+1),temp_u(fluidnum,n,k,j,i-1),n,i);
              }

              if(i==32||i==33||i==34||i==35||i==36){
                fprintf(stdout,"dul=%11.7e,dur=%11.7e,uc=%11.7e,n=%4i,i=%4i \n",dul(fluidnum,n,i),dur(fluidnum,n,i),uc(fluidnum,n,i),n,i);
              }
            }
          }
        }



        if (pmb->precon->uniform_limiter[X1DIR]) {
          fprintf(stdout,"van Leer\n");
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il-1; i<=iu; ++i) {
                du2(fluidnum,i) = dul(fluidnum,n,i)*dur(fluidnum,n,i);
                dum(fluidnum,n,i) = 2.0*du2(fluidnum,i)/(dul(fluidnum,n,i) + dur(fluidnum,n,i));
                if (du2(fluidnum,i) <= 0.0) dum(fluidnum,n,i) = 0.0;

                if(i==32||i==33||i==34||i==35){
                  fprintf(stdout,"du2=%11.7e,dum=%11.7e,n=%4i,i=%4i\n",du2(fluidnum,i),dum(fluidnum,n,i),n,i);
                }
              }
            }
          }
        } else {
          fprintf(stdout,"MUSCL\n");
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il-1; i<=iu; ++i) {
                du2(fluidnum,i) = dul(fluidnum,n,i)*dur(fluidnum,n,i);
                if(dul(fluidnum,n,i)<0.0){
                  sign1=-1.0;
                }else{
                  sign1=1.0;
                }
                if(dur(fluidnum,n,i)<0.0){
                  sign2=-1.0;
                }else{
                  sign2=1.0;
                }
                if((dul(fluidnum,n,i)+dur(fluidnum,n,i))<0.0){
                  sign3=-1.0;
                }else{
                  sign3=1.0;
                }

                min1 = std::min(0.25*std::abs(dul(fluidnum,n,i)+dur(fluidnum,n,i)),std::abs(dul(fluidnum,n,i)));
                if(min1==0.25*std::abs(dul(fluidnum,n,i)+dur(fluidnum,n,i))){
                  min1=min1*sign3;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign3\n");
                  }
                }else if(min1==std::abs(dul(fluidnum,n,i))){
                  min1=min1*sign1;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign1\n");
                  }
                }

                if(min1<0.0){
                  sign4=-1.0;
                }else{
                  sign4=1.0;
                }

                truemin = std::min(std::abs(min1),std::abs(dur(fluidnum,n,i)));

                if(truemin==std::abs(dur(fluidnum,n,i))){
                  truemin=truemin*sign2;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign2\n");
                  }
                }else if(truemin==std::abs(min1)){
                  truemin=truemin*sign4;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign4\n");
                  }
                }
             
                dum(fluidnum,n,i) = 2.0*truemin;
                if (du2(fluidnum,i) <= 0.0) dum(fluidnum,n,i) = 0.0;
                if(i==12||i==13||i==14||i==15||i==16){
                  fprintf(stdout,"du2=%11.7e,dum=%11.7e,n=%4i,i=%4i\n",du2(fluidnum,i),dum(fluidnum,n,i),n,i);
                }
              }
            }
          }
        } 

//SUPER BEE OPTION!!! new flags? change uniform_limiter[] to have options, not just bool flag?
//    } else {
//      fprintf(stdout,"super bee\n");
//      for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//        for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
//#pragma omp simd simdlen(SIMD_WIDTH)
//          for (int i=il-1; i<=iu; ++i) {
//             du2(fluidnum,i) = dul(fluidnum,n,i)*dur(fluidnum,n,i);
//             dum(fluidnum,n,i) = std::max(std::min(2.0*dul(fluidnum,n,i),dur(fluidnum,n,i)),
//                 std::min(dul(fluidnum,n,i),2.0*dur(fluidnum,n,i)));
//             if (du2(fluidnum,i) <= 0.0) dum(fluidnum,n,i) = 0.0;
//          }
//        }
//      }
//    }

//end superbee


        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
            for (int i=il-1; i<=iu; ++i) {
              ul(fluidnum,n,k,j,i+1) = uc(fluidnum,n,i) + ((pco->x1f(i+1)-pco->x1v(i))/pco->dx1f(i))*dum(fluidnum,n,i);
              ur(fluidnum,n,k,j,i  ) = uc(fluidnum,n,i) - ((pco->x1v(i  )-pco->x1f(i))/pco->dx1f(i))*dum(fluidnum,n,i);
          
              if(i==32||i==33||i==34||i==35||i==36){
                fprintf(stdout,"ur=%11.7e,ul=%11.7e,uc=%11.7e,n=%4i,i=%4i\n",ur(fluidnum,n,k,j,i),ul(fluidnum,n,k,j,i+1),uc(fluidnum,n,i),n,i);
              }

            }
       
          }
        }

//        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//#pragma omp simd simdlen(SIMD_WIDTH)
//          for (int i=il-1; i<=iu; ++i) {
//            ur(fluidnum,1,k,j,i)=ur(fluidnum,1,k,j,i)*ur(fluidnum,0,k,j,i);
//            ul(fluidnum,1,k,j,i)=ul(fluidnum,1,k,j,i)*ul(fluidnum,0,k,j,i);
//            ur(fluidnum,2,k,j,i)=ur(fluidnum,2,k,j,i)*ur(fluidnum,0,k,j,i);
//            ul(fluidnum,2,k,j,i)=ul(fluidnum,2,k,j,i)*ul(fluidnum,0,k,j,i);
//            ur(fluidnum,3,k,j,i)=ur(fluidnum,3,k,j,i)*ur(fluidnum,0,k,j,i);
//            ul(fluidnum,3,k,j,i)=ul(fluidnum,3,k,j,i)*ul(fluidnum,0,k,j,i);
//          }
//        }
       
       
      }
    }
    pmb->peos->ConservedToPrimitive(ul,w,pfield->b,wl,pfield->bcc,pco,il-1,iu,jl,ju,kl,ku);
    pmb->peos->ConservedToPrimitive(ur,w,pfield->b,wr,pfield->bcc,pco,il-1,iu,jl,ju,kl,ku);

    for(int n=0; n<(NWAVE+NINT+NSCALARS); ++n){
      for(int i=il-1;i<=iu;++i){
        if(i==32||i==33||i==34||i==35||i==36){
          fprintf(stdout,"after floors: wl=%11.3e,wr=%11.3e,n=%4i,i=%4i\n",wl(0,n,0,9,i),wr(0,n,0,9,i),n,i);
        }
      }
    }

  }else{

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
    // compute L/R slopes for each variable
        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NHYDRO); ++n) {
#pragma omp simd
            for (int i=il-1; i<=iu; ++i) {
              dwl(fluidnum,n,i) = (w(fluidnum,n,k,j,i  ) - w(fluidnum,n,k,j,i-1));
              dwr(fluidnum,n,i) = (w(fluidnum,n,k,j,i+1) - w(fluidnum,n,k,j,i  ));
              wc(fluidnum,n,i) = w(fluidnum,n,k,j,i);

              if(i==32||i==33||i==34||i==65||i==66){
                fprintf(stdout,"w=%11.7e,w+1=%11.7e,w-1=%11.7e,n=%4i,i=%4i\n",w(fluidnum,n,k,j,i),w(fluidnum,n,k,j,i+1),w(fluidnum,n,k,j,i-1),n,i);
              }

#ifdef DEBUG_ALL
              if(i==32||i==33||i==34||i==65||i==66){
                fprintf(stdout,"i=%4i, w left=%13.7e  w mid=%13.7e  w right=%13.7e\n",i,w(fluidnum,n,k,j,i-1),w(fluidnum,n,k,j,i),w(fluidnum,n,k,j,i+1));
                fprintf(stdout,"PLM! i=%4i f%4i, n=%4i, dwl=%13.7e, dwr=%13.7e, wc=%13.7e\n",i,fluidnum,n,dwl(fluidnum,n,i),dwr(fluidnum,n,i),wc(fluidnum,n,i));
#endif
              }
            }
          }
        }
        if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
          for (int i=il-1; i<=iu; ++i) {
            bx(i) = bcc(IB1,k,j,i);

            dwl(IBY,i) = (bcc(IB2,k,j,i  ) - bcc(IB2,k,j,i-1));
            dwr(IBY,i) = (bcc(IB2,k,j,i+1) - bcc(IB2,k,j,i  ));
            wc(IBY,i) = bcc(IB2,k,j,i);

            dwl(IBZ,i) = (bcc(IB3,k,j,i  ) - bcc(IB3,k,j,i-1));
            dwr(IBZ,i) = (bcc(IB3,k,j,i+1) - bcc(IB3,k,j,i  ));
            wc(IBZ,i) = bcc(IB3,k,j,i);
          }
        }

    // Project slopes to characteristic variables, if necessary
    // Note order of characteristic fields in output vect corresponds to (IVX,IVY,IVZ)
        if (pmb->precon->characteristic_reconstruction) {
          LeftEigenmatrixDotVector(pmb,IVX,il-1,iu,bx,wc,dwl);
          LeftEigenmatrixDotVector(pmb,IVX,il-1,iu,bx,wc,dwr);
        }

    // Internal energy and advected scalars
    // It looks like since dwl, dwr are not addressed for n>=NWAVE in LEDV, 
    // they are kept identical, consistent with Athena comment.

    // Apply van Leer limiter for uniform grid
        if (pmb->precon->uniform_limiter[X1DIR]) {
          fprintf(stdout,"van Leer\n");
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il-1; i<=iu; ++i) {
                dw2(fluidnum,i) = dwl(fluidnum,n,i)*dwr(fluidnum,n,i);
                dwm(fluidnum,n,i) = 2.0*dw2(fluidnum,i)/(dwl(fluidnum,n,i) + dwr(fluidnum,n,i));
                if (dw2(fluidnum,i) <= 0.0) dwm(fluidnum,n,i) = 0.0;
              }
            }
          }
    // Apply Mignone limiter for non-uniform grid
        } else {
          fprintf(stdout,"Mignone\n");
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il-1; i<=iu; ++i) {
                dw2(fluidnum,i) = dwl(fluidnum,n,i)*dwr(fluidnum,n,i);
                Real cf = pco->dx1v(i  )/(pco->x1f(i+1) - pco->x1v(i));
                Real cb = pco->dx1v(i-1)/(pco->x1v(i  ) - pco->x1f(i));
                dwm(fluidnum,n,i) = (dw2(fluidnum,i)*(cf*dwl(fluidnum,n,i) + cb*dwr(fluidnum,n,i))/
                  (SQR(dwl(fluidnum,n,i)) + SQR(dwr(fluidnum,n,i)) + dw2(fluidnum,i)*(cf + cb - 2.0)));
                if (dw2(fluidnum,i) <= 0.0) dwm(fluidnum,n,i) = 0.0;
                if(i==32||i==33||i==34||i==65||i==66){
                  fprintf(stdout,"dw2=%11.7e,dwm=%11.7e,n=%4i,i=%4i\n",dw2(fluidnum,i),dwm(fluidnum,n,i),n,i);
                }
              }
            }
          }
        }
  

    // Project limited slope back to primitive variables, if necessary
        if (pmb->precon->characteristic_reconstruction) {
          RightEigenmatrixDotVector(pmb,IVX,il-1,iu,bx,wc,dwm);
        }

    // Internal energy and advected scalars

    // compute ql_(i+1/2) and qr_(i-1/2) using monotonized slopes
        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
            for (int i=il-1; i<=iu; ++i) {
              wl(fluidnum,n,k,j,i+1) = wc(fluidnum,n,i) + ((pco->x1f(i+1)-pco->x1v(i))/pco->dx1f(i))*dwm(fluidnum,n,i);
              wr(fluidnum,n,k,j,i  ) = wc(fluidnum,n,i) - ((pco->x1v(i  )-pco->x1f(i))/pco->dx1f(i))*dwm(fluidnum,n,i);
         
#ifdef DEBUG_ALL
              fprintf(stdout,"pre-floors: wl=%11.7e,wr=%11.7e,n=%4i,i=%4i\n",wl(fluidnum,n,k,j,i),wr(fluidnum,n,k,j,i),n,i);
#endif

              if (pmb->precon->characteristic_reconstruction) {
          // Reapply EOS floors to both L/R reconstructed primitive states
                pmb->peos->ApplyPrimitiveFloors(wl, fluidnum, k, j, i+1);
                pmb->peos->ApplyPrimitiveFloors(wr, fluidnum, k, j, i);
              }
            }
          }
        }

      }
    }

  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX2()
//  \brief

void Reconstruction::PiecewiseLinearX2(MeshBlock *pmb,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  Coordinates *pco = pmb->pcoord;
  Field *pfield;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> bx,dw2,wc,dwl,dwr,dwm;
  AthenaArray<Real> temp_u,ul,ur,du2,uc,dul,dur,dum;

  temp_u.InitWithShallowCopy(pmb->precon->scr15_ni_);
  ul.InitWithShallowCopy(pmb->precon->scr16_ni_);
  ur.InitWithShallowCopy(pmb->precon->scr17_ni_);
  du2.InitWithShallowCopy(pmb->precon->scr02_i_);
  uc.InitWithShallowCopy(pmb->precon->scr1_ni_);
  dul.InitWithShallowCopy(pmb->precon->scr2_ni_);
  dur.InitWithShallowCopy(pmb->precon->scr3_ni_);
  dum.InitWithShallowCopy(pmb->precon->scr4_ni_);
  bx.InitWithShallowCopy(pmb->precon->scr01_i_);
  dw2.InitWithShallowCopy(pmb->precon->scr02_i_);
  wc.InitWithShallowCopy(pmb->precon->scr1_ni_);
  dwl.InitWithShallowCopy(pmb->precon->scr2_ni_);
  dwr.InitWithShallowCopy(pmb->precon->scr3_ni_);
  dwm.InitWithShallowCopy(pmb->precon->scr4_ni_);

  Real truemin, min1, trigger, iden, sign1, sign2, sign3, sign4;

  if (pmb->precon->cons_reconstruction){

    pmb->peos->PrimitiveToConserved(w,pfield->bcc,temp_u,pco,il-1,iu,jl-1,ju,kl-1,ku);

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {

//        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//#pragma omp simd simdlen(SIMD_WIDTH)
//          for (int i=il-1; i<=iu; ++i) {
//            iden=1.0/temp_u(fluidnum,0,k,j,i);
//            temp_u(fluidnum,1,k,j,i)=temp_u(fluidnum,1,k,j,i)*iden;
//            temp_u(fluidnum,2,k,j,i)=temp_u(fluidnum,2,k,j,i)*iden;
//            temp_u(fluidnum,3,k,j,i)=temp_u(fluidnum,3,k,j,i)*iden;
//          }
//        }

        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NHYDRO); ++n) {
#pragma omp simd
            for (int i=il-1; i<=iu; ++i) {
              dul(fluidnum,n,i) = (temp_u(fluidnum,n,k,j  ,i) - temp_u(fluidnum,n,k,j-1,i));
              dur(fluidnum,n,i) = (temp_u(fluidnum,n,k,j+1,i) - temp_u(fluidnum,n,k,j  ,i));
              uc(fluidnum,n,i) = temp_u(fluidnum,n,k,j,i);

              if(i==32||i==33||i==34||i==35||i==36){
                fprintf(stdout,"w=%11.7e,w+1=%11.7e,w-1=%11.7e,n=%4i,j=%4i\n",w(fluidnum,n,k,j,i),w(fluidnum,n,k,j+1,i),w(fluidnum,n,k,j-1,i),n,j);
                fprintf(stdout,"u=%11.7e,u+1=%11.7e,u-1=%11.7e,n=%4i,j=%4i\n",temp_u(fluidnum,n,k,j,i),temp_u(fluidnum,n,k,j+1,i),temp_u(fluidnum,n,k,j-1,i),n,j);
              }

              if(i==32||i==33||i==34||i==35||i==36){
                fprintf(stdout,"dul=%11.7e,dur=%11.7e,uc=%11.7e,n=%4i,i=%4i \n",dul(fluidnum,n,i),dur(fluidnum,n,i),uc(fluidnum,n,i),n,i);
              }
            }
          }
        }



        if (pmb->precon->uniform_limiter[X1DIR]) {
          fprintf(stdout,"van Leer\n");
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il-1; i<=iu; ++i) {
                du2(fluidnum,i) = dul(fluidnum,n,i)*dur(fluidnum,n,i);
                dum(fluidnum,n,i) = 2.0*du2(fluidnum,i)/(dul(fluidnum,n,i) + dur(fluidnum,n,i));
                if (du2(fluidnum,i) <= 0.0) dum(fluidnum,n,i) = 0.0;

                if(i==2||i==3||i==4||i==5){
                  fprintf(stdout,"du2=%11.7e,dum=%11.7e,n=%4i,i=%4i\n",du2(fluidnum,i),dum(fluidnum,n,i),n,i);
                }
              }
            }
          }
        } else {
          fprintf(stdout,"MUSCL\n");
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il-1; i<=iu; ++i) {
                du2(fluidnum,i) = dul(fluidnum,n,i)*dur(fluidnum,n,i);
                if(dul(fluidnum,n,i)<0.0){
                  sign1=-1.0;
                }else{
                  sign1=1.0;
                }
                if(dur(fluidnum,n,i)<0.0){
                  sign2=-1.0;
                }else{
                  sign2=1.0;
                }
                if((dul(fluidnum,n,i)+dur(fluidnum,n,i))<0.0){
                  sign3=-1.0;
                }else{
                  sign3=1.0;
                }

                min1 = std::min(0.25*std::abs(dul(fluidnum,n,i)+dur(fluidnum,n,i)),std::abs(dul(fluidnum,n,i)));
                if(min1==0.25*std::abs(dul(fluidnum,n,i)+dur(fluidnum,n,i))){
                  min1=min1*sign3;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign3\n");
                  }
                }else if(min1==std::abs(dul(fluidnum,n,i))){
                  min1=min1*sign1;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign1\n");
                  }
                }

                if(min1<0.0){
                  sign4=-1.0;
                }else{
                  sign4=1.0;
                }
                truemin = std::min(std::abs(min1),std::abs(dur(fluidnum,n,i)));

                if(truemin==std::abs(dur(fluidnum,n,i))){
                  truemin=truemin*sign2;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign2\n");
                  }
                }else if(truemin==std::abs(min1)){
                  truemin=truemin*sign4;
                  if(i==12||i==13||i==14||i==15||i==16){
                    fprintf(stdout,"sign4\n");
                  }
                }

                dum(fluidnum,n,i) = 2.0*truemin;
                if (du2(fluidnum,i) <= 0.0) dum(fluidnum,n,i) = 0.0;
                if(i==12||i==13||i==14||i==15||i==16){
                  fprintf(stdout,"du2=%11.7e,dum=%11.7e,n=%4i,i=%4i\n",du2(fluidnum,i),dum(fluidnum,n,i),n,i);
                }
              }
            }
          }
        }
        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
            for (int i=il-1; i<=iu; ++i) {
              ul(fluidnum,n,k,j+1,i) = uc(fluidnum,n,i) + ((pco->x1f(j+1)-pco->x1v(j))/pco->dx1f(j))*dum(fluidnum,n,i);
              ur(fluidnum,n,k,j  ,i) = uc(fluidnum,n,i) - ((pco->x1v(j  )-pco->x1f(j))/pco->dx1f(j))*dum(fluidnum,n,i);

              if(i==32||i==33||i==34||i==35||i==36){
                fprintf(stdout,"ur=%11.7e,ul=%11.7e,uc=%11.7e,n=%4i,i=%4i\n",ur(fluidnum,n,k,j,i),ul(fluidnum,n,k,j+1,i),uc(fluidnum,n,i),n,j);
              }

            }

          }
        }

//        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
//#pragma omp simd simdlen(SIMD_WIDTH)
//          for (int i=il-1; i<=iu; ++i) {
//            ur(fluidnum,1,k,j,i)=ur(fluidnum,1,k,j,i)*ur(fluidnum,0,k,j,i);
//            ul(fluidnum,1,k,j,i)=ul(fluidnum,1,k,j,i)*ul(fluidnum,0,k,j,i);
//            ur(fluidnum,2,k,j,i)=ur(fluidnum,2,k,j,i)*ur(fluidnum,0,k,j,i);
//            ul(fluidnum,2,k,j,i)=ul(fluidnum,2,k,j,i)*ul(fluidnum,0,k,j,i);
//            ur(fluidnum,3,k,j,i)=ur(fluidnum,3,k,j,i)*ur(fluidnum,0,k,j,i);
//            ul(fluidnum,3,k,j,i)=ul(fluidnum,3,k,j,i)*ul(fluidnum,0,k,j,i);
//          }
//        }


    
      }
    }

    pmb->peos->ConservedToPrimitive(ul,w,pfield->b,wl,pfield->bcc,pco,il,iu,jl,ju,kl,ku);
    pmb->peos->ConservedToPrimitive(ur,w,pfield->b,wr,pfield->bcc,pco,il,iu,jl,ju,kl,ku);

    for(int n=0; n<(NWAVE+NINT+NSCALARS); ++n){
      for(int i=il-1;i<=iu;++i){
        if(i==32||i==33||i==34||i==35||i==36){
          fprintf(stdout,"after floors: wl=%11.3e,wr=%11.3e,n=%4i,i=%4i\n",wl(0,n,0,9,i),wr(0,n,0,9,i),n,i);
        }
      }
    }


  }else{

//////////////////////////////////////////////
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl-1; j<=ju; ++j) {
    // compute L/R slopes for each variable
        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NHYDRO); ++n) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              dwl(fluidnum,n,i) = (w(fluidnum,n,k,j  ,i) - w(fluidnum,n,k,j-1,i));
              dwr(fluidnum,n,i) = (w(fluidnum,n,k,j+1,i) - w(fluidnum,n,k,j  ,i));
              wc(fluidnum,n,i) = w(fluidnum,n,k,j,i);
            }
          }
        }
    

        if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            bx(i) = bcc(IB2,k,j,i);

            dwl(IBY,i) = (bcc(IB3,k,j  ,i) - bcc(IB3,k,j-1,i));
            dwr(IBY,i) = (bcc(IB3,k,j+1,i) - bcc(IB3,k,j  ,i));
            wc(IBY,i) = bcc(IB3,k,j,i);

            dwl(IBZ,i) = (bcc(IB1,k,j  ,i) - bcc(IB1,k,j-1,i));
            dwr(IBZ,i) = (bcc(IB1,k,j+1,i) - bcc(IB1,k,j  ,i));
            wc(IBZ,i) = bcc(IB1,k,j,i);
          }
        }

    // Project slopes to characteristic variables, if necessary
    // Note order of characteristic fields in output vect corresponds to (IVY,IVZ,IVX)
        if (pmb->precon->characteristic_reconstruction) {
          LeftEigenmatrixDotVector(pmb,IVY,il,iu,bx,wc,dwl);
          LeftEigenmatrixDotVector(pmb,IVY,il,iu,bx,wc,dwr);
        }

    // Apply van Leer limiter for uniform grid
        if (pmb->precon->uniform_limiter[X2DIR]) {
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il; i<=iu; ++i) {
                dw2(fluidnum,i) = dwl(fluidnum,n,i)*dwr(fluidnum,n,i);
                dwm(fluidnum,n,i) = 2.0*dw2(fluidnum,i)/(dwl(fluidnum,n,i) + dwr(fluidnum,n,i));
                if (dw2(fluidnum,i) <= 0.0) dwm(fluidnum,n,i) = 0.0;
              }
            }
          }
      

    // Apply Mignone limiter for non-uniform grid
        }else{
          for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
            for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
              for (int i=il; i<=iu; ++i) {
                dw2(fluidnum,i) = dwl(fluidnum,n,i)*dwr(fluidnum,n,i);
                Real cf = pco->dx2v(j  )/(pco->x2f(j+1) - pco->x2v(j));
                Real cb = pco->dx2v(j-1)/(pco->x2v(j  ) - pco->x2f(j));
                dwm(fluidnum,n,i) = (dw2(fluidnum,i)*(cf*dwl(fluidnum,n,i) + cb*dwr(fluidnum,n,i))/
                  (SQR(dwl(fluidnum,n,i)) + SQR(dwr(fluidnum,n,i)) + dw2(fluidnum,i)*(cf + cb - 2.0)));
                if (dw2(fluidnum,i) <= 0.0) dwm(fluidnum,n,i) = 0.0;
              }
            }
          }
      
        }

    // Project limited slope back to primitive variables, if necessary
        if (pmb->precon->characteristic_reconstruction) {
          RightEigenmatrixDotVector(pmb,IVY,il,iu,bx,wc,dwm);
        }

    // compute ql_(j+1/2) and qr_(j-1/2) using monotonized slopes
        for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
          for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
            for (int i=il; i<=iu; ++i) {
              wl(fluidnum,n,k,j+1,i) = wc(fluidnum,n,i) + ((pco->x2f(j+1)-pco->x2v(j))/pco->dx2f(j))*dwm(fluidnum,n,i);
              wr(fluidnum,n,k,j  ,i) = wc(fluidnum,n,i) - ((pco->x2v(j  )-pco->x2f(j))/pco->dx2f(j))*dwm(fluidnum,n,i);
         
              if (pmb->precon->characteristic_reconstruction) {
          // Reapply EOS floors to both L/R reconstructed primitive states
                pmb->peos->ApplyPrimitiveFloors(wl, fluidnum, k, j+1, i);
                pmb->peos->ApplyPrimitiveFloors(wr, fluidnum, k, j, i);
              }
            }
          }
        }
      }
    }
  }
    return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::PiecewiseLinearX3()
//  \brief

void Reconstruction::PiecewiseLinearX3(MeshBlock *pmb,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &w, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &wl, AthenaArray<Real> &wr) {
  Coordinates *pco = pmb->pcoord;
  // set work arrays to shallow copies of scratch arrays
  AthenaArray<Real> bx,dw2,wc,dwl,dwr,dwm;
  bx.InitWithShallowCopy(pmb->precon->scr01_i_);
  dw2.InitWithShallowCopy(pmb->precon->scr02_i_);
  wc.InitWithShallowCopy(pmb->precon->scr1_ni_);
  dwl.InitWithShallowCopy(pmb->precon->scr2_ni_);
  dwr.InitWithShallowCopy(pmb->precon->scr3_ni_);
  dwm.InitWithShallowCopy(pmb->precon->scr4_ni_);

  for (int k=kl-1; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
    // compute L/R slopes for each variable
    for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
      for (int n=0; n<(NHYDRO); ++n) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
            dwl(fluidnum,n,i) = (w(fluidnum,n,k  ,j,i) - w(fluidnum,n,k-1,j,i));
            dwr(fluidnum,n,i) = (w(fluidnum,n,k+1,j,i) - w(fluidnum,n,k  ,j,i));
            wc(fluidnum,n,i) = w(fluidnum,n,k,j,i);
          }
        }}
    if (MAGNETIC_FIELDS_ENABLED) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        bx(i) = bcc(IB3,k,j,i);

        dwl(IBY,i) = (bcc(IB1,k  ,j,i) - bcc(IB1,k-1,j,i));
        dwr(IBY,i) = (bcc(IB1,k+1,j,i) - bcc(IB1,k  ,j,i));
        wc(IBY,i) = bcc(IB1,k,j,i);

        dwl(IBZ,i) = (bcc(IB2,k  ,j,i) - bcc(IB2,k-1,j,i));
        dwr(IBZ,i) = (bcc(IB2,k+1,j,i) - bcc(IB2,k  ,j,i));
        wc(IBZ,i) = bcc(IB2,k,j,i);
      }
    }

    // Project slopes to characteristic variables, if necessary
    // Note order of characteristic fields in output vect corresponds to (IVZ,IVX,IVY)
    if (pmb->precon->characteristic_reconstruction) {
      LeftEigenmatrixDotVector(pmb,IVZ,il,iu,bx,wc,dwl);
      LeftEigenmatrixDotVector(pmb,IVZ,il,iu,bx,wc,dwr);
    }


    // Apply van Leer limiter for uniform grid
    if (pmb->precon->uniform_limiter[X3DIR]) {
      for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
        for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
          for (int i=il; i<=iu; ++i) {
              dw2(fluidnum,i) = dwl(fluidnum,n,i)*dwr(fluidnum,n,i);
              dwm(fluidnum,n,i) = 2.0*dw2(fluidnum,i)/(dwl(fluidnum,n,i) + dwr(fluidnum,n,i));
              if (dw2(fluidnum,i) <= 0.0) dwm(fluidnum,n,i) = 0.0;
            }
          }
        }
      

    // Apply Mignone limiter for non-uniform grid
    } else {
      for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
        for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
          for (int i=il; i<=iu; ++i) {
              dw2(fluidnum,i) = dwl(fluidnum,n,i)*dwr(fluidnum,n,i);
              Real cf = pco->dx3v(k  )/(pco->x3f(k+1) - pco->x3v(k));
              Real cb = pco->dx3v(k-1)/(pco->x3v(k  ) - pco->x3f(k));
              dwm(fluidnum,n,i) = (dw2(fluidnum,i)*(cf*dwl(fluidnum,n,i) + cb*dwr(fluidnum,n,i))/
                (SQR(dwl(fluidnum,n,i)) + SQR(dwr(fluidnum,n,i)) + dw2(fluidnum,i)*(cf + cb - 2.0)));
              if (dw2(fluidnum,i) <= 0.0) dwm(fluidnum,n,i) = 0.0;
            }
          }
        }
      }
    

    // Project limited slope back to primitive variables, if necessary
    if (pmb->precon->characteristic_reconstruction) {
      RightEigenmatrixDotVector(pmb,IVZ,il,iu,bx,wc,dwm);
    }

    // compute ql_(k+1/2) and qr_(k-1/2) using monotonized slopes
    for(int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
      for (int n=0; n<(NWAVE+NINT+NSCALARS); ++n) {
#pragma omp simd simdlen(SIMD_WIDTH)
        for (int i=il; i<=iu; ++i) {
            wl(fluidnum,n,k+1,j,i) = wc(fluidnum,n,i) + ((pco->x3f(k+1)-pco->x3v(k))/pco->dx3f(k))*dwm(fluidnum,n,i);
            wr(fluidnum,n,k  ,j,i) = wc(fluidnum,n,i) - ((pco->x3v(k  )-pco->x3f(k))/pco->dx3f(k))*dwm(fluidnum,n,i);
          
        if (pmb->precon->characteristic_reconstruction) {
          // Reapply EOS floors to both L/R reconstructed primitive states
          pmb->peos->ApplyPrimitiveFloors(wl, fluidnum, k+1, j, i);
          pmb->peos->ApplyPrimitiveFloors(wr, fluidnum, k, j, i);
        }
      }}
    }
  }}

  return;
}
