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
#include <algorithm>  // sort
#include <cfloat>     // FLT_MAX
#include <cmath>      // std::abs(), pow()
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept> 

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "comoving.hpp"

//Constructor for comoving object which will actually make changes
Comoving::Comoving(MeshBlock *pmb, ParameterInput *pin) {
  //LockPos = pin->GetOrAddReal("mesh","cmR0",0.0); //Get initial scalar position
  //LockVel = pin->GetOrAddReal("mesh","cmV0",0.0); //Get initial scalar velocity
  if (COORDINATE_SYSTEM == "cartesian") {
    CoordSystem = 1;
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    CoordSystem = 2;
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    CoordSystem = 3;
  } else {
    CoordSystem = 0; 
    std::cout << "### Warning in Comoving creator" << std::endl
        << "Coordinate System " << COORDINATE_SYSTEM << " not valid coordinate system for a comoving frame" << std::endl;
     
  }

  GridStage = 1;
  std::string integrator = pin->GetOrAddString("time","integrator","vl2");
  if (integrator == "vl2"){ 
    nstages = 2;
  } else if (integrator == "rk2"){
    nstages = 2;
  } else if (integrator == "rk3"){
    nstages = 3;
  } else if (integrator == "rk4"){
    nstages = 4;
  } else if (integrator == "ssprk5_4"){
    nstages = 5;
  } else {
    nstages = 0;
    std::cout << "### Warning in Comoving creator"
           << "integrator=" << integrator << " not valid time integrator" << std::endl;
  
  }

  MeshBlock *pmy_block = pmb;
  int is, ie, js, je, ks, ke, ng;
  if (pmb->pcoord->coarse_flag==true) {
    is = pmb->cis; js = pmb->cjs; ks = pmb->cks;
    ie = pmb->cie; je = pmb->cje; ke = pmb->cke;
    ng=pmb->cnghost;
  } else {
    is = pmb->is; js = pmb->js; ks = pmb->ks;
    ie = pmb->ie; je = pmb->je; ke = pmb->ke;
    ng=NGHOST;
  }
  Mesh *pm=pmy_block->pmy_mesh;
    
  //get initial grid from meshBlock, and make correctly sized x1f, x2f, x3f   
  int ncells1 = (ie-is+1) + 2*ng;
  int ncells2 = 1, ncells3 = 1;
  if (pmb->block_size.nx2 > 1) ncells2 = (je-js+1) + 2*ng;
  if (pmb->block_size.nx3 > 1) ncells3 = (ke-ks+1) + 2*ng;
  
  delx1f.NewAthenaArray(ncells1);
  delx2f.NewAthenaArray(ncells2); 
  delx3f.NewAthenaArray(ncells3);
  
  a1f.NewAthenaArray(ncells1);
  a2f.NewAthenaArray(ncells2);
  a3f.NewAthenaArray(ncells3);
  
  x1fi.InitWithShallowCopy(pmb->pcoord->x1f);
  x2fi.InitWithShallowCopy(pmb->pcoord->x2f);
  x3fi.InitWithShallowCopy(pmb->pcoord->x3f);

  


}


//destructor
Comoving::~Comoving(){


}

//void Comoving::UpdateComovingLock(Mesh *pm, int stage){
//  if (pm->CMLocking != NULL) {
//    pm->CMLocking(pm,gvx1f,gvx2f,gvx3f);    
//  }
//  GridStage++;
//  GridStage%(nstages+1);
//  
//  
//}







//void Comoving::UpdateGrid(Mesh *pm, int stage){
//  //std::cout << stage << std::endl;
//  //Edit Region data in Mesh
//  //Edit all MeshBlock Data
//  //Edit coord object
//  pm->EditGrid(delx1f,delx2f,delx3f);  
//
//}


void Comoving::ComovingSrcTerms(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){


}



//AthenaArray<Real> Comoving::ShockDetector(Mesh *pm){


//}

