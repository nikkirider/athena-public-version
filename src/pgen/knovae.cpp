//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file knovae.cpp
//  \brief Problem generator for spherical knova  problem.  
//
//

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

//====================================================================================
// global variables
FILE *otffile;

//====================================================================================
// local functions
Real LogMeshSpacingX1(Real x, RegionSize rs);

void ReflectInnerX1_nonuniform(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

//====================================================================================
// Enroll user-specific functions
void Mesh::InitUserMeshData(ParameterInput *pin) {
  Real x1rat = pin->GetOrAddReal("mesh","x1rat",0.0);
  
  if (x1rat < 0.0) {
    EnrollUserMeshGenerator(X1DIR, LogMeshSpacingX1);
    if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(INNER_X1, ReflectInnerX1_nonuniform);
    }
  }
  return;
}

//====================================================================================
// "Logarithmic" (power-law) mesh
// Note that the grid setup asks for the **local** x1min, x1max etc. 
// x is the "logical" position in the grid, with the logical grid runing
// from 0 to 1, i.e. x = i/Nx
Real LogMeshSpacingX1(Real x, RegionSize rs) {
  Real xf, xrat;
  xrat   = pow(rs.x1max/rs.x1min,1.0/((Real) rs.nx1)); // Only valid for fixed grid, no MPI
  xf     = rs.x1min*pow(xrat,x*rs.nx1); // x = i/Nx
  return xf;
}

//========================================================================================
// Reflecting inner X1 boundary conditions for radially non-uniform grids

void ReflectInnerX1_nonuniform(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones, reflecting v1
  for (int n=0; n<(NHYDRO); ++n) {
    if (n==(IVX)) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(IVX,k,j,is-i) = -prim(IVX,k,j,(is+i-1));  // reflect 1-velocity
          }
        }
      }
    } else {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            prim(n,k,j,is-i) = prim(n,k,j,(is+i-1));
          }
        }
      }
    }
  }
  // copy face-centered magnetic fields into ghost zones, reflecting b1
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x1f(k,j,(is-i)) = -b.x1f(k,j,(is+i  ));  // reflect 1-field
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k,j,(is-i)) =  b.x2f(k,j,(is+i-1));
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          b.x3f(k,j,(is-i)) =  b.x3f(k,j,(is+i-1));
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real b0=0.0, angle=0.0;
  Real pi         = 4.0*std::atan(1.0);
  Real r0         = pin->GetReal("problem","r0"); // radius of initial ejecta
  Real dr         = pin->GetReal("problem","dr"); // width of transition
  Real p0         = pin->GetReal("problem","p0"); // ambient pressure
  Real d0         = pin->GetReal("problem","d0"); // ambient density
  Real v0         = pin->GetReal("problem","v0"); // velocity of ejecta
  Real m0         = pin->GetReal("problem","m0"); // mass of ejecta
  Real E0         = pin->GetReal("problem","E0"); // total energy of ejecta
  int  ihomol     = pin->GetOrAddInteger("problem","ihomol",0); // initial homologous expansion (no "ring")
  Real thopen     = pin->GetOrAddReal("problem","thopen",pi); // opening angle of disk ejecta. pi is full polar coverage
  Real rwindtidev = pin->GetOrAddReal("problem","rwindtidev",1.0); // ratio between wind and tidal ejecta speed.
  Real rwindtiden = pin->GetOrAddReal("problem","rwindtiden",1.0); // ratio between wind and tidal ejecta density.
  int  ilog       = pin->GetOrAddInteger("mesh", "ilog", 0);
  Real x1min      = pin->GetReal("mesh","x1min");
  if ((!ihomol) && (x1min >= 0.5*r0)) { // for ring, make sure that x1min < r0/2
    std::stringstream msg;
    msg << "### FATAL ERROR in knovae.cpp ProblemGenerator" << std::endl
        << "x1min > 0.5*r0 " << x1min << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    b0    = pin->GetReal("problem","b0");
    angle = (PI/180.0)*pin->GetReal("problem","angle");
  }
  Real gamma   = peos->GetGamma();
  Real gm1     = gamma - 1.0;
  Real d1      = 3.0*m0/(4.0*pi*pow(r0,3));
  Real e1      = 3.0*E0/(4.0*pi*pow(r0,3));
  Real e0      = p0/gm1;

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x1_0   = pin->GetOrAddReal("problem","x1_0",0.0);
  Real x2_0   = pin->GetOrAddReal("problem","x2_0",0.0);
  Real x3_0   = pin->GetOrAddReal("problem","x3_0",0.0);
  Real x0,y0,z0;
  if (COORDINATE_SYSTEM == "cartesian") {
    x0 = x1_0;
    y0 = x2_0;
    z0 = x3_0;
  } else if (COORDINATE_SYSTEM == "cylindrical") {
    x0 = x1_0*std::cos(x2_0);
    y0 = x1_0*std::sin(x2_0);
    z0 = x3_0;
  } else if (COORDINATE_SYSTEM == "spherical_polar") {
    x0 = x1_0*std::sin(x2_0)*std::cos(x3_0);
    y0 = x1_0*std::sin(x2_0)*std::sin(x3_0);
    z0 = x1_0*std::cos(x2_0);
  } else {
    // Only check legality of COORDINATE_SYSTEM once in this function
    std::stringstream msg;
    msg << "### FATAL ERROR in knovae.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // setup uniform ambient medium with spherical over-pressured region
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real rad,v0rad,radv,thet;
        if (COORDINATE_SYSTEM == "cartesian") {
          Real x   = pcoord->x1v(i);
          Real y   = pcoord->x2v(j);
          Real z   = pcoord->x3v(k);
          rad      = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          thet   = std::acos(z/rad);
        } else if (COORDINATE_SYSTEM == "cylindrical") {
          Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
          Real z = pcoord->x3v(k);
          rad    = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          thet   = std::acos(z/rad);
        } else { // if (COORDINATE_SYSTEM == "spherical_polar")
          Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
          Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
          Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
          rad    = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          thet   = pcoord->x2v(j);
        }

        // factors for radial and polar dependence
        Real rho2              = d1;
        Real rho1              = rho2*rwindtiden;
        Real rho0              = d0;
        Real fm1rad            = 0.5*(1.0-std::tanh((rad-r0)/dr)); // radial drop off
        Real fm1phi            = 0.25*((1.0+std::tanh((thet-0.5*(pi-thopen))/0.1))*(1.0-std::tanh((thet-0.5*(pi+thopen))/0.1))); // polar drop off
        if (ilog) {
          radv = (rad-x1min)/(r0-x1min);
        } else {
          radv = rad;
        }
        if (ihomol) { // ramp to make ring
          v0rad = v0*rad/r0;
        } else {
          if (rad <= 0.5*r0) {
            v0rad         = 2.0*v0*rad/r0;
          } else {
            v0rad         = v0;
          }
        }
        //phydro->u(IDN,k,j,i) = d0+(d1-d0)*fm1rad*fm1phi;
        phydro->u(IDN,k,j,i) = rho0+(rho1-rho0)*fm1rad+(rho2-rho1)*fm1phi*fm1rad; // that correct?
        phydro->u(IM1,k,j,i) = (v0rad*(1.0-rwindtidev)*fm1rad*fm1phi+v0rad*rwindtidev*fm1rad)*phydro->u(IDN,k,j,i);
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          //phydro->u(IEN,k,j,i) = p0/gm1 + 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) = e0+(e1-e0)*0.5*(1.0-std::tanh((rad-r0)/dr));
          if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
            phydro->u(IEN,k,j,i) += d0;
        }
        if (DUAL_ENERGY) {
          phydro->u(IIE,k,j,i) = e0+(e1-e0)*0.5*(1.0-std::tanh((rad-r0)/dr));
        }
        if (NSCALARS == 2) {
          // the first index corresponds to tidal ejecta, the second to wind ejecta
          phydro->u(NHYDRO-NSCALARS  ,k,j,i) = rho2*fm1rad*fm1phi;       // tidal ejecta
          phydro->u(NHYDRO-NSCALARS+1,k,j,i) = rho1*fm1rad*(1.0-fm1phi); // wind ejecta
        }
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x1f(k,j,i) = b0 * std::cos(angle);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x1f(k,j,i) =
                b0 * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x1f(k,j,i) = b0 * std::abs(std::sin(theta))
                * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            pfield->b.x2f(k,j,i) = b0 * std::sin(angle);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real phi = pcoord->x2v(j);
            pfield->b.x2f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real theta = pcoord->x2v(j);
            Real phi = pcoord->x3v(k);
            pfield->b.x2f(k,j,i) = b0 * std::cos(theta)
                * (std::cos(angle) * std::cos(phi) + std::sin(angle) * std::sin(phi));
            if (std::sin(theta) < 0.0)
              pfield->b.x2f(k,j,i) *= -1.0;
          }
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "cylindrical") {
            pfield->b.x3f(k,j,i) = 0.0;
          } else { //if (COORDINATE_SYSTEM == "spherical_polar") {
            Real phi = pcoord->x3v(k);
            pfield->b.x3f(k,j,i) =
                b0 * (std::sin(angle) * std::cos(phi) - std::cos(angle) * std::sin(phi));
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*b0*b0;
        }
      }
    }
  }

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign("knovae.otf");
    std::stringstream msg;
    if ((otffile = fopen(fname.c_str(),"wb")) == NULL) {
      msg << "### FATAL ERROR in function [Meshblock::ProblemGenerator]"
          << std::endl << "knovae.otf could not be opened" <<std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }

}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief otf diagnostics (shell position, sphericity etc)
//========================================================================================
void MeshBlock::UserWorkInLoop(void) {
#ifdef MPI_PARALLEL
  int mpierr;
#endif
  if (NSCALARS == 2) {
    // Average shock position based on scalar. Assumes center at origin.
    int nscl=2,nsum=3; // number of scalars. 
    int nelt=nsum*(nscl+1); // quantities to calculate (rad, thet, mass)
    Real 
    
    Real *inbuf  = (Real*) calloc(nelt,sizeof(Real));
    for (int l=0; l<nelt; l++) {
      inbuf[l] = 0.0;
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real rad, thet;
          Real dv = pcoord->GetCellVolume(k,j,i);
          if (COORDINATE_SYSTEM == "cartesian") {
            Real x   = pcoord->x1v(i);
            Real y   = pcoord->x2v(j);
            Real z   = pcoord->x3v(k);
            rad      = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            thet     = std::acos(z/rad);
          } else if (COORDINATE_SYSTEM == "cylindrical") {
            Real x = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j));
            Real z = pcoord->x3v(k);
            rad    = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            thet   = std::acos(z/rad);
          } else { // if (COORDINATE_SYSTEM == "spherical_polar")
            Real x = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::cos(pcoord->x3v(k));
            Real y = pcoord->x1v(i)*std::sin(pcoord->x2v(j))*std::sin(pcoord->x3v(k));
            Real z = pcoord->x1v(i)*std::cos(pcoord->x2v(j));
            rad    = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            thet   = pcoord->x2v(j);
          }
          for (int l=0; l<nscl; l++) {  // for the individual tracers
            Real s = phydro->u(NHYDRO-NSCALARS+l,k,j,i);
            inbuf[nsum*l+0] += rad *s*dv; // for the radius
            inbuf[nsum*l+1] += thet*s*dv; // for theta
            inbuf[nsum*l+2] += s*dv;      // total mass
          }
          Real d = phydro->u(NHYDRO-NSCALARS,k,j,i)+phydro->u(NHYDRO-NSCALARS+1,k,j,i); // total
          inbuf[nsum*nscl+0] += rad *d*dv;
          inbuf[nsum*nscl+1] += thet*d*dv;
          inbuf[nsum*nscl+2] += d*dv;
        }
      }
    }
#ifdef MPI_PARALLEL
    mpierr = MPI_Allreduce(MPI_IN_PLACE,inbuf,nelt,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
    //if (mpierr) {
    //  msg << "[MeshBlock::UserWorkInLoop]: MPI_Allreduce error = "  << mpierr << std::endl;
    //  throw std::runtime_error(msg.str().c_str());
    //}
    //for (int l=0; l<nsum*(nscl+1); l++) inbuf[l] = outbuf[l];
#endif // MPI_PARALLEL
    for (int l=0; l<=nscl; l++) {
      inbuf[nsum*l+0] /= inbuf[nsum*l+2]; // divide radius by mass, for each tracer l
      inbuf[nsum*l+1] /= inbuf[nsum*l+2]; // divide theta by mass, for each tracer l
    }
    // only the root process writes to file 
    if (Globals::my_rank == 0) {
      Real *data = (Real*) calloc(nelt+1,sizeof(Real));
      data[0] = pmy_mesh->time;
      for (int l=0; l<nelt; l++) {
        data[l+1] = inbuf[l];
      }
      fwrite(data,sizeof(Real),nelt+1,otffile); 
      for (int l=0; l<=nelt; l++) {
        std::cout << std::setprecision(5) << data[l] << ' ';
      }
      std::cout << std::endl;
      free(data);
    }
    free(inbuf);
  } // if (NSCALARS == 2)
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (Globals::my_rank == 0) {
    fclose(otffile);
  }
  return;
}
