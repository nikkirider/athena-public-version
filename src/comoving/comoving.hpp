#ifndef COMOVING_HPP_
#define COMOVING_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file comoving.hpp
//  \brief definitions for Comoving Class
// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

class Mesh;
class MeshBlock;
class ParameterInput;
class Hydro;
class Coordinates;


class Comoving {
public:
  friend class Mesh;
  //Constructors and Destructor
  Comoving(Mesh* pm, ParameterInput* pin);
  //Comoving(Mesh* pm, ParameterInput* pin, Real null_flag);
  ~Comoving();
  
  //Scalar location and velocity
  Real LockPos;
  Real LockVel;
  int  GridStage;
  int  CoordSystem;
  AthenaArray<Real> delx1f; //Difference from old grid to new grid in each coordinate direction
  AthenaArray<Real> delx2f;
  AthenaArray<Real> delx3f;
  AthenaArray<Real> gvx1f; //Velocity of each cell wall
  AthenaArray<Real> gvx2f;
  AthenaArray<Real> gvx3f;
  AthenaArray<Real> gridVel;


  //LockingFunction_t CMLocking;  
  void UpdateComovingLock(Mesh *pm, int stage);
  //void EnrollComovingLockingFunction(LockingFunction_t myfunc);
  void UpdateGrid(Mesh *pm, int stage);
  //void ComovingSrcTerms(Hydro *phydro, ParameterInput *pin, int stage);
  void ComovingSrcTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
  
private:
  //AthenaArray<Real> ShockDetector(Mesh *pm);

};


#endif // COMOVING_HPP_
