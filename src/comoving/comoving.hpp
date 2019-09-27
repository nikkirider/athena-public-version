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
class ParameterInput;
class Hydro;
class Coordinates;


class Comoving {
public:
  //Constructors and Destructor
  Comoving(Mesh* pm, ParameterInput* pin);
  Comoving(Mesh* pm, ParameterInput* pin, Real null_flag);
  ~Comoving();
  
  //Scalar location and velocity
  Real LockPos;
  Real LockVel;
  int  GridStage;
  int  CoordSystem;
  AthenaArray<Real> Zeta;
  
  void UpdateLockedData(Mesh *pm, int SCALAR);
  void UpdateGrid(Mesh *pm);
  void ComovingSrcTerms(Hydro *phydro, ParameterInput *pin);

private:
  AthenaArray<Real> ShockDetector(Mesh *pm);

};


#endif // COMOVING_HPP_
