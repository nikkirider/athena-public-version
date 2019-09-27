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

//Constructor for comoving object which will actually make changes
Comoving::Comoving(Mesh *pm, ParameterInput *pin) {
  LockPos = pin->GetReal("problem","r0"); //Get initial scalar position
  LockVel = pin->GetReal("problem","v0"); //Get initial scalar velocity
  if (COORDINATE_SYSTEM == "cartesian") {
    CoordSystem = 1;
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    CoordSystem = 2;
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    CoordSystem = 3;
  } else {
    CoordSystem = 0;  
  }
  
}

//Null Comoving object which does not do anything.
Comoving::Comoving(Mesh *pm, ParameterInput *pin, Real null_flag){
  LockPos = 0.0;
}

//destructor
Comoving::~Comoving(){


}

//
void Comoving::UpdateLockedData(Mesh *pm, int SCALAR){


}



void Comoving::UpdateGrid(Mesh *pm){
  //Edit Region data in Mesh
  //Edit all MeshBlock Data
  //Edit coord object

}


void Comoving::ComovingSrcTerms(Hydro *phydro,ParameterInput *pin){


}



AthenaArray<Real> Comoving::ShockDetector(Mesh *pm){


}

