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
  //friend class Mesh;
  friend class Coordinates;
  friend class MeshBlock;
  //Constructors and Destructor
  Comoving(MeshBlock* pmb, ParameterInput *pin);
  //Comoving(Mesh* pm, ParameterInput* pin, Real null_flag);
  ~Comoving();
  
  //Scalar location and velocity
  //Real InitGridPos;
  int  GridStage;
  int  CoordSystem;
  int  nstages;
  Real x10;
  Real x20;
  Real x30;
  AthenaArray<Real> delx1f, delx2f, delx3f; //Difference from old grid to new grid in each coordinate direction
  AthenaArray<Real> a1f, a2f, a3f; //Velocity of each cell wall

  //LockingFunction_t CMLocking;  
  //void UpdateComovingLock(Mesh *pm, int stage);
  //void EnrollComovingLockingFunction(LockingFunction_t myfunc);
  //void UpdateGrid(Mesh *pm, int stage);
  

  //void EditDelta(Real delta,int ind, int dir, Comoving &cm);
  void EditCoordObj(MeshBlock *pmb,Coordinates *pcoord);

  //void ComovingSrcTerms(Hydro *phydro, ParameterInput *pin, int stage);
  void ComovingSrcTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
  
private:
  //AthenaArray<Real> ShockDetector(Mesh *pm);
  AthenaArray<Real> x1fi, x2fi, x3fi;
};


#endif // COMOVING_HPP_
