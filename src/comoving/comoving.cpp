//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file comoving.cpp
//  \brief implementation of Comoving class.
//
//
// C/C++ headers
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "comoving.hpp"

//Constructor
Comoving::Comoving(Mesh *pm, ParameterInput *pin) {
  ShockPos = pin->GetReal("problem","r0"); //Set initial Radius 
  
}

Comoving::Comoving(Mesh *pm, ParameterInput *pin, Real null_flag){
  ShockPos = 0.0;
}

//destructor
Comoving::~Comoving(){


}

