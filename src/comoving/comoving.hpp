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
  Real ShockPos;
  Real ShockVel;
  Comoving(Mesh* pm, ParameterInput* pin);
  Comoving(Mesh* pm, ParameterInput* pin, Real null_flag);
  ~Comoving();
private:
  void ShockDetector(Mesh *pm);

};


#endif // COMOVING_HPP_
