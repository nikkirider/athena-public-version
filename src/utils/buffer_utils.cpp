//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffer_utils.cpp
//  \brief namespace containing buffer utilities.

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "buffer_utils.hpp"

namespace BufferUtility {

//----------------------------------------------------------------------------------------
//! \fn void Pack5DData(AthenaArray<Real> &src, Real *buf, int sn, int en,
//                     int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//  \brief pack a 5D AthenaArray into a one-dimensional buffer

void Pack5DData(AthenaArray<Real> &src, Real *buf, int sn, int en,
                int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
  for (int n=sn; n<=en; ++n) {
    for (int k=sk; k<=ek; k++) {
      for (int j=sj; j<=ej; j++) {
#pragma omp simd
        for (int i=si; i<=ei; i++){
            buf[offset++]=src(fluidnum,n,k,j,i);
        }
      }
    }
  }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Unpack5DData(Real *buf, AthenaArray<Real> &dst, int sn, int en,
//                        int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//  \brief unpack a one-dimensional buffer into a 5D AthenaArray

void Unpack5DData(Real *buf, AthenaArray<Real> &dst, int sn, int en,
                  int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
  for (int n=sn; n<=en; ++n) {
    for (int k=sk; k<=ek; ++k) {
      for (int j=sj; j<=ej; ++j) {
#pragma omp simd
        for (int i=si; i<=ei; ++i){
          dst(fluidnum,n,k,j,i) = buf[offset++];
        }
      }
    }
  }
  }
  return;
}

//----------------------------------------------------------------------------------------
////! \fn void Pack4DData(AthenaArray<Real> &src, Real *buf, int sn, int en,
////                     int si, int ei, int sj, int ej, int sk, int ek, int &offset)
////  \brief pack a 4D AthenaArray into a one-dimensional buffer
//
void Pack4DData(AthenaArray<Real> &src, Real *buf, int sn, int en,
         int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
  for (int k=sk; k<=ek; k++) {
    for (int j=sj; j<=ej; j++) {
#pragma omp simd
        for (int i=si; i<=ei; i++){
          buf[offset++]=src(fluidnum,k,j,i);
        }
     }
  }
  }
  return;
}

//----------------------------------------------------------------------------------------
////! \fn void Unpack4DData(Real *buf, AthenaArray<Real> &dst, int sn, int en,
////                        int si, int ei, int sj, int ej, int sk, int ek, int &offset)
////  \brief unpack a one-dimensional buffer into a 4D AthenaArray
//
void Unpack4DData(Real *buf, AthenaArray<Real> &dst, int sn, int en,
            int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int fluidnum=0;fluidnum<(NFLUIDS);fluidnum++){
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma omp simd
      for (int i=si; i<=ei; ++i)
        dst(fluidnum,k,j,i) = buf[offset++];
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Pack3DData(AthenaArray<Real> &src, Real *buf,
//                      int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//  \brief pack a 3D AthenaArray into a one-dimensional buffer

void Pack3DData(AthenaArray<Real> &src, Real *buf,
                int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int k=sk; k<=ek; k++) {
    for (int j=sj; j<=ej; j++) {
#pragma omp simd
      for (int i=si; i<=ei; i++)
          buf[offset++]=src(k, j, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Unpack3DData(Real *buf, AthenaArray<Real> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int &offset)
//  \brief unpack a one-dimensional buffer into a 3D AthenaArray

void Unpack3DData(Real *buf, AthenaArray<Real> &dst,
                  int si, int ei, int sj, int ej, int sk, int ek, int &offset) {
  for (int k=sk; k<=ek; ++k) {
    for (int j=sj; j<=ej; ++j) {
#pragma omp simd
      for (int i=si; i<=ei; ++i)
        dst(k,j,i) = buf[offset++];
    }
  }
  return;
}

} // end namespace BufferUtility
