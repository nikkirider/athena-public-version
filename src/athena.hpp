#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

// C headers
#include <math.h>
#include <stdint.h>  // int64_t

// C++ headers

// Athena++ headers
#include "athena_arrays.hpp"
#include "defs.hpp"

// typedefs that allow code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED
  typedef float Real;
  #ifdef MPI_PARALLEL
  #define MPI_ATHENA_REAL MPI_FLOAT
  #endif
#else
  typedef double Real;
  #ifdef MPI_PARALLEL
  #define MPI_ATHENA_REAL MPI_DOUBLE
  #endif
#endif

// for OpenMP 4.0 SIMD vectorization, control width of SIMD lanes
#if defined(__AVX512F__)
#define SIMD_WIDTH 8
#elif defined(__AVX__)
#define SIMD_WIDTH 4
#elif defined(__SSE2__)
#define SIMD_WIDTH 2
#else
#define SIMD_WIDTH 4
#endif

#define CACHELINE_BYTES 64

class MeshBlock;
class Coordinates;
class ParameterInput;
struct RegionSize;
class HydroDiffusion;
class FieldDiffusion;

class Mesh;
//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of meshblock

typedef struct LogicalLocation {
  // These values can exceed the range of int32_t if >= 30 levels of AMR are used
  int64_t lx1, lx2, lx3;
  int level;

  LogicalLocation() : lx1(-1), lx2(-1), lx3(-1), level(-1) {}

  // operators useful for sorting
  bool operator==(LogicalLocation &ll)
    { return ((ll.level==level) && (ll.lx1==lx1) && (ll.lx2==lx2) && (ll.lx3==lx3)); }
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right)
    { return left.level < right.level; };
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right)
    { return left.level > right.level; };

} LogicalLocation;


//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

typedef struct RegionSize {
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of x(i)/x(i-1)
  // the size of the root grid or a MeshBlock should not exceed int32_t limits
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
} RegionSize;

//----------------------------------------------------------------------------------------
//! \struct OTFData
//  \brief on-the-fly data structure
typedef struct OTFData {
  int64_t len;
  Real *data;
} OTFData;

//---------------------------------------------------------------------------------------
//! \struct FaceField
//  \brief container for face-centered fields

typedef struct FaceField {
  AthenaArray<Real> x1f,x2f,x3f;
} FaceField;

//----------------------------------------------------------------------------------------
//! \struct EdgeField
//  \brief container for edge-centered fields

typedef struct EdgeField {
  AthenaArray<Real> x1e,x2e,x3e;
} EdgeField;

//----------------------------------------------------------------------------------------
// enums used everywhere

// array indices for conserved: density, momemtum, total energy, internal energy,
// face-centered field
enum {IDN=0, IM1=1, IM2=2, IM3=3, IEN=4, IIE=5};
enum {IB1=0, IB2=1, IB3=2};

// array indices for cless variables
enum {IE11=4,IE22=5,IE33=6,IE12=7,IE13=8,IE23=9};

// array indices for 1D primitives: velocity, pressure internal energy, transverse components of field
enum {IVX=1, IVY=2, IVZ=3, IPR=4, IGE=5, IBY=(NHYDRO), IBZ=((NHYDRO)+1)};

// array indices for transverse component of field to be used for vectors of length NWAVE
enum {IWBY=(NWAVE-2), IWBZ=(NWAVE-1)};

// array indices for cless variables
enum {IP11=4,IP22=5,IP33=6,IP12=7,IP13=8,IP23=9};

// array indices for face-centered electric fields returned by Riemann solver
enum {X1E2=0, X1E3=1, X2E3=0, X2E1=1, X3E1=0, X3E2=1};

// array indices for passive scalars (only goes to 10), this is mostly for pgen 
enum {IS0=(NHYDRO-NSCALARS)  , IS1=(NHYDRO-NSCALARS)+1, IS2=(NHYDRO-NSCALARS)+2,
			IS3=(NHYDRO-NSCALARS)+3, IS4=(NHYDRO-NSCALARS)+4, IS5=(NHYDRO-NSCALARS)+5,
			IS6=(NHYDRO-NSCALARS)+6, IS7=(NHYDRO-NSCALARS)+7, IS8=(NHYDRO-NSCALARS)+8,
			IS9=(NHYDRO-NSCALARS)+9};

// array indices for metric and triangular matrices in GR
enum {I00, I01, I02, I03, I11, I12, I13, I22, I23, I33, NMETRIC};
enum {T00, T10, T11, T20, T21, T22, T30, T31, T32, T33, NTRIANGULAR};

// needed for arrays dimensioned over grid directions
enum CoordinateDirection {X1DIR=0, X2DIR=1, X3DIR=2};

// needed wherever MPI communications are used.  Must be < 32 and unique
enum Athena_MPI_Tag {TAG_HYDRO=0, TAG_FIELD=1, TAG_RAD=2, TAG_CHEM=3, TAG_HYDFLX=4,
  TAG_FLDFLX=5, TAG_RADFLX=6, TAG_CHMFLX=7, TAG_AMR=8, TAG_FLDFLX_POLE=9, TAG_GRAVITY=11,
  TAG_MGGRAV=12,TAG_SHBOX_HYDRO=13,TAG_SHBOX_FIELD=14,TAG_SHBOX_EMF=15, TAG_CLESS=16, 
	TAG_CLFLX=17};

enum BoundaryType {BNDRY_HYDRO=0, BNDRY_FIELD=1, BNDRY_GRAVITY=2, BNDRY_MGGRAV=3,
                   BNDRY_MGGRAVF=4, BNDRY_FLCOR=5, BNDRY_EMFCOR=6, BNDRY_CLESS=7,
									 BNDRY_FLCORCL=8};
enum CCBoundaryType {HYDRO_CONS=0, HYDRO_PRIM=1, CLESS_CONS=2, CLESS_PRIM=3};
enum FluxCorrectionType {FLUX_HYDRO=0, FLUX_CLESS=1};

//----------------------------------------------------------------------------------------
// function pointer prototypes for user-defined modules set at runtime

typedef void (*BValFunc_t)(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                           FaceField &b, Real time, Real dt,
                           int is, int ie, int js, int je, int ks, int ke, int ngh);
typedef void (*BValFuncCL_t)(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke); // cless-BC 
typedef int (*AMRFlagFunc_t)(MeshBlock *pmb);
typedef Real (*MeshGenFunc_t)(Real x, RegionSize rs);
typedef void (*SrcTermFunc_t)(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
typedef Real (*StaticGravPotFunc_t)(const Real x1, const Real x2, const Real x3, const Real time);
typedef Real (*TimeStepFunc_t)(MeshBlock *pmb);
typedef Real (*HistoryOutputFunc_t)(MeshBlock *pmb, int iout);
typedef void (*MetricFunc_t)(Real x1, Real x2, Real x3, ParameterInput *pin,
             AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
             AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
typedef void (*MGBoundaryFunc_t)(AthenaArray<Real> &dst,Real time, int nvar,
             int is, int ie, int js, int je, int ks, int ke, int ngh,
             Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
typedef void (*GravityBoundaryFunc_t)(MeshBlock *pmb, Coordinates *pco,
             AthenaArray<Real> &dst, Real time, Real dt,
             int is, int ie, int js, int je, int ks, int ke);
typedef void (*ViscosityCoeff_t)(HydroDiffusion *phdif, MeshBlock *pmb,
             const  AthenaArray<Real> &w, const AthenaArray<Real> &bc,
             int is, int ie, int js, int je, int ks, int ke);
typedef void (*ConductionCoeff_t)(HydroDiffusion *phdif, MeshBlock *pmb,
              const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
              int is, int ie, int js, int je, int ks, int ke);
typedef void (*FieldDiffusionCoeff_t)(FieldDiffusion *pfdif, MeshBlock *pmb,
                                      const AthenaArray<Real> &w,
                                      const AthenaArray<Real> &bmag,
                                      int is, int ie, int js, int je, int ks, int ke);

typedef Real (*WallVel_t)(Real xf, int i, Real time, Real dt, int dir, AthenaArray<Real> gridData);
typedef void (*CalcGridData_t)(Mesh *pm);
#endif // ATHENA_HPP_
