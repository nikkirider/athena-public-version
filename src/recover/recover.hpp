#ifndef RECOVER_RECOVER_HPP_
#define RECOVER_RECOVER_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recover.hpp
//  \brief definitions for Recover class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
//#include "../task_list/task_list.hpp"

class MeshBlock;
class ParameterInput;
class Coordinates;

//! \class Recover
//  \brief Recovery mechanism to repeat timestep.

class Recover {

  friend class Field;
  friend class Hydro;
  friend class Mesh;

public:

  Recover(MeshBlock *pmb, ParameterInput *pin);
  ~Recover();

  // Old Data: full copy of old  set
  AthenaArray<Real> u,w;   // primitive variables
  FaceField b;  // face-centered magnetic fields
  AthenaArray<Real> x1f, x2f, x3f; // wall coordinates, from which grid will be reconstructed 
  AthenaArray<Real> phi; // gravitational potential

  Real time_old, dt_old;
  int il, iu, jl, ju, kl, ku, ng; //With Ghost cells
  int ie,is,je,js,ke,ks; //Without ghost cells

  // initializes recovery system (i.e. copies initial data into backup)
  void Initialize(MeshBlock *pmb); 
  // checks the meshblock pmb, resets to old values and adjusts freduct.
  bool Check(MeshBlock *pmb);
  void Reset(MeshBlock *pmb, bool failed); // reset if failed, and adjust freduct_
  int ReduceDtFactor() { return freduct_; };

private:
  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Recover

protected:
  int freduct_; // reduction factor. 0: no reduction

};
#endif // RECOVER_RECOVER_HPP_
