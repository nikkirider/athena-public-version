//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recover.cpp
//  \brief implementation of Recover class
//    The class Recover is owned by MeshBlock, similar to Expansion.
// C/C++ headers
#include <string>
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()
// Athena++ headers
#include "../globals.hpp"
#include "recover.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../reconstruct/reconstruction.hpp"
// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//============================================
// constructor, initializes data structures and parameters
// Needs to be called after grid setup, problem generator, and timestep.
//============================================
Recover::Recover(MeshBlock *pmb, ParameterInput *pin) {
  bool coarse_flag=pmb->pcoord->CoarseFlag();
  pmy_block = pmb;
  if (coarse_flag==true) {
    is = pmb->cis; js = pmb->cjs; ks = pmb->cks;
    ie = pmb->cie; je = pmb->cje; ke = pmb->cke;
    ng=pmb->cnghost;
  } else {
    is = pmb->is; js = pmb->js; ks = pmb->ks;
    ie = pmb->ie; je = pmb->je; ke = pmb->ke;
    ng=NGHOST;
  }

  // Allocate memory for mesh dimensions
  int ncells1 = (ie-is+1) + 2*ng;
  il = is-ng;
  iu = ie+ng;
  int ncells2 = 1, ncells3 = 1;
  jl = js;
  ju = je;
  kl = ks;
  ku = ke;
  if (pmb->block_size.nx2 > 1) {ncells2 = (je-js+1) + 2*ng; jl = js-ng; ju = je+ng;}
  if (pmb->block_size.nx3 > 1) {ncells3 = (ke-ks+1) + 2*ng; kl = ks-ng; ku = ke+ng;}

  if (EXPANDING) { // Allocate grid data if needed
    x1f.NewAthenaArray((ncells1+1));
    x2f.NewAthenaArray((ncells2+1));
    x3f.NewAthenaArray((ncells3+1));
  }

  // allocate conservative and primitive variables (old values)
  u.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);
  w.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);

  // allocate magnetic fields if needed
  if (MAGNETIC_FIELDS_ENABLED) {
    int ncells1_b = pmb->block_size.nx1 + 2*(NGHOST);
    int ncells2_b = 1, ncells3_b = 1;
    if (pmb->block_size.nx2 > 1) ncells2_b = pmb->block_size.nx2 + 2*(NGHOST);
    if (pmb->block_size.nx3 > 1) ncells3_b = pmb->block_size.nx3 + 2*(NGHOST);

    // Note the extra cell in each longitudinal dirn for interface fields
    b.x1f.NewAthenaArray( ncells3_b   , ncells2_b   ,(ncells1_b+1));
    b.x2f.NewAthenaArray( ncells3_b   ,(ncells2_b+1), ncells1_b   );
    b.x3f.NewAthenaArray((ncells3_b+1), ncells2_b   , ncells1_b   );
  }
    
  if (EXPANDING) { // initialize grid information
#pragma omp simd
    for (int i=il; i<=iu+1;++i)
      x1f(i) = pmb->pcoord->x1f(i);
#pragma omp simd
    for (int j=jl; j<=ju+1;++j)
     x2f(j) = pmb->pcoord->x2f(j);
#pragma omp simd
    for (int k=kl; k<=ku+1;++k)
      x3f(k) = pmb->pcoord->x3f(k);
  }

  // copy conservative and primitive variables
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          u(n,k,j,i) = pmb->phydro->u(n,k,j,i);
          w(n,k,j,i) = pmb->phydro->w(n,k,j,i);       
        }
      }
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu+1; ++i) {
          b.x1f(k,j,i) = pmb->pfield->b.x1f(k,j,i);
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x2f(k,j,i) = pmb->pfield->b.x2f(k,j,i);
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          b.x3f(k,j,i) = pmb->pfield->b.x3f(k,j,i);
        }
      }
    }
  }

  if (SELF_GRAVITY_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          phi(k,j,i) = pmb->pgrav->phi(k,j,i);
        }
      }
    }
  }

  dt_old  = pmb->pmy_mesh->dt; 
  freduct_ = 0; // reduction level 0: no reduction. 

}


//============================================
// Destructor
//============================================
Recover::~Recover() {

  if (EXPANDING) {
    x1f.DeleteAthenaArray();
    x2f.DeleteAthenaArray();
    x3f.DeleteAthenaArray();
  }
  u.DeleteAthenaArray();
  w.DeleteAthenaArray();
  if (MAGNETIC_FIELDS_ENABLED) {
    b.x1f.DeleteAthenaArray();
    b.x2f.DeleteAthenaArray();
    b.x3f.DeleteAthenaArray();
  }
  if (SELF_GRAVITY_ENABLED) {
    phi.DeleteAthenaArray();
  }
}

//============================================
// Recover::Check(MeshBlock *pmb) 
// \brief: Checks grid owned by pmb for invalid values.
//   Called by Mesh::CheckAndReset()
//============================================
bool Recover::Check(MeshBlock *pmb) {

  bool failed = false;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real dens = pmb->phydro->u(IDN,k,j,i);
        Real etot = pmb->phydro->u(IEN,k,j,i);
        failed = (failed || (dens <=0.0) || (std::isnan(dens))
                         || (etot <=0.0) || (std::isnan(etot)));
        if (DUAL_ENERGY) {
          Real eint = pmb->phydro->u(IIE,k,j,i);;
          failed = (failed || (eint <=0.0) || (std::isnan(eint)));
        }
      }
    }
  }
  return failed;
}

//============================================
// Recover::Resets(MeshBlock *pmb) 
// // \brief: If failed, resets primitive variables,
// otherwise copies new primitive variables into 
// temporary. This is called after BCs, therefore
// must include boundaries.
//============================================
void Recover::Reset(MeshBlock *pmb, bool failed) {

  if (failed) {
    // copy backup into primitive variables
    for (int n=0; n<NHYDRO; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->phydro->w(n,k,j,i) = w(n,k,j,i); 
            pmb->phydro->u(n,k,j,i) = u(n,k,j,i);
          }
        }
      }
    }

    if (EXPANDING) { // grids
#pragma omp simd
      for (int i=il; i<=iu+1;++i)
        pmb->pcoord->x1f(i) = x1f(i);
#pragma omp simd
      for (int j=jl; j<=ju+1;++j) 
        pmb->pcoord->x2f(j) = x2f(j);
#pragma omp simd
      for (int k=kl; k<=ku+1;++k) 
        pmb->pcoord->x3f(k) = x3f(k);
    }

    if (MAGNETIC_FIELDS_ENABLED) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd 
          for (int i=il; i<=iu+1; ++i) {
            pmb->pfield->b.x1f(k,j,i) = b.x1f(k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pfield->b.x2f(k,j,i) = b.x2f(k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pfield->b.x3f(k,j,i) = b.x3f(k,j,i);
          }
        }
      }
    } 

    if (SELF_GRAVITY_ENABLED) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pgrav->phi(k,j,i) = phi(k,j,i);
          }
        }
      }
    }

  } else {
    // copy new conservative and primitive variables into backup
    for (int n=0; n<NHYDRO; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            w(n,k,j,i) = pmb->phydro->w(n,k,j,i);
            u(n,k,j,i) = pmb->phydro->u(n,k,j,i);
          }
        }
      }
    }

    if (EXPANDING) { // grids
#pragma omp simd
      for (int i=il; i<=iu+1;++i)
        x1f(i) = pmb->pcoord->x1f(i);
#pragma omp simd
      for (int j=jl; j<=ju+1;++j)
        x2f(j) = pmb->pcoord->x2f(j);
#pragma omp simd
      for (int k=kl; k<=ku+1;++k)
        x3f(k) = pmb->pcoord->x3f(k);
    }

    if (MAGNETIC_FIELDS_ENABLED) { 
      for (int k=kl; k<=ku; ++k) { 
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd 
          for (int i=il; i<=iu+1; ++i) {
            b.x1f(k,j,i) = pmb->pfield->b.x1f(k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            b.x2f(k,j,i) = pmb->pfield->b.x2f(k,j,i);
          }
        }
      }
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            b.x3f(k,j,i) = pmb->pfield->b.x3f(k,j,i);
          }
        }
      }
    }

    if (SELF_GRAVITY_ENABLED) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            phi(k,j,i) = pmb->pgrav->phi(k,j,i);
          }
        }
      }
    }

  } // if (failed)

  return;
}


