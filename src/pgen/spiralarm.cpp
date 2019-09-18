//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file spiralarm.cpp
//  \brief Problem generator for spiral arm problem.  
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
// local functions
Real gravpot_spirarm(const Real x1, const Real x2, const Real x3, const Real time);
void SpiralInflowInnerX1(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int is, int ie, int js, int je, int ks, int ke, int ngh);
void SpiralOutflowOuterX1(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int is, int ie, int js, int je, int ks, int ke, int ngh);
Real GetMeanDensity(Coordinates *pcoord, Hydro *phydro);

static void stop_this();

//====================================================================================
// Enroll user-specific functions
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // vertical external gravitational potential
  EnrollStaticGravPotFunction(gravpot_spirarm); // Cox & Gomez

  // enroll user-defined boundary conditions
  if (mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X1, SpiralInflowInnerX1);
  }
  //if (mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
  //  EnrollUserBoundaryFunction(OUTER_X1, SpiralOutflowOuterX1);
  //}

  Real four_pi_G = 48.0*std::atan(1.0); // unit system (n,T) with G=1
  Real eps = 0.0;
  SetFourPiG(four_pi_G);
  SetGravityThreshold(eps);
  SetMeanDensity(0.0); // temporary -- will be set to mean density in initialization
  return;
}

//====================================================================================
// short for debugging interrupt
static void stop_this() {
  std::stringstream msg;
  msg << "stop" << std::endl;
  throw std::runtime_error(msg.str().c_str());
}

//====================================================================================
// Real GetMeanDensity(void)
Real GetMeanDensity(Coordinates *pcoord, Hydro *phydro) {
  MeshBlock* pmb = phydro->pmy_block;
  std::stringstream msg;
#ifdef MPI_PARALLEL
  int  mpierr;
#endif
  Real mass[2], gmass[2];
  mass[0] = 0.0; mass[1] = 0.0;
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        Real dv  = pcoord->GetCellVolume(k,j,i);
        mass[0] += phydro->u(IDN,k,j,i)*dv;
        mass[1] += dv;
      }
    }
  }
#ifdef MPI_PARALLEL
  mpierr = MPI_Allreduce(&mass, &gmass, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[GetMeanDensity]: MPI_Allreduce error = "
        << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int n=0; n<2; n++) mass[n] = gmass[n];
#endif // MPI_PARALLEL
  mass[0] /= mass[1];
  return mass[0];
}

//====================================================================================
//====================================================================================
// Begin class Potential: container for all variables etc to initialize and calculate 
// potential. Initialized in problem generator, after reading problem-specific 
// variables.
//====================================================================================
class Potential {
  public:
    Potential(const int iprob, const int itemp, const Real T0, const Real n0, 
              const Real vx0, const Real rhos0, const Real ramp, const Real x1len,
              const Real zmin, const Real zmax, const Real awarp, const int kwarp,
              const int nx2, const Real gam);

    // wrapper for StaticGravPotFunc
    Real gravpot(const Real x1, const Real x2, const Real x3, const Real time);
    // access
    Real T0() {return _T0;};
    Real vx0() {return _vx0;};
    Real ramp() {return _ramp;};
    Real Teq(const Real n);
    Real dens(const int k) {return _dens(k);}; // Needed by problem setup (half-arrays)
    Real temp(const int k) {return _temp(k);};
    void temperature(); // Sets the half-array temperature array after density integration.
    // The following functions interface with OdeIntegrator. Their names
    // must be exactly as used here.
    Real dydx(const Real z,const Real n); 
    int  nx() {return _nx2;};
    Real xmin() {return _zmin;}; 
    Real xmax() {return _zmax;};
    Real x(const int k) {return _zhaf(k);};
    Real y0() {return _n0;};
    AthenaArray<Real> &y = _dens; // presumably this makes a copy of the pointer.
    // End interfaces.
  
  private:
    int  _iprob;
    int  _itemp;
    int  _nx2;
    Real _n0;
    Real _T0;
    Real _P0;
    Real _vx0;
    Real _ramp;
    Real _x1len;
    Real _x2len;
    Real _dx2;
    Real _zmin;
    Real _zmax;
    Real _awarp;
    Real _gam;
    Real _rgc;
    Real _rhos0;
    int  _kwarp;
    const Real conv_pc   = 8.87652670e+00;
    const Real conv_kmsc = 9.11836593e-02;
    static const int  _nord     = 3;
    const int  _narm     = 2;
    const Real _H0       = 180.0/conv_pc;
    const Real _r0       = 3e3/conv_pc;
    const Real _rs       = 3e3/conv_pc;
    const Real _sa       = sin(20*PI/180.0);
    const Real _ta       = tan(20*PI/180.0);
    const Real _Cn[_nord]= {8.0/(3.0*PI),0.5,8.0/(15.0*PI)};
    const Real _a1       = 6.5e3/conv_pc;
    const Real _b1       = 0.26e3/conv_pc;
    const Real _C1       = 8.887e3/conv_pc;
    const Real _a2       = 0.7e3/conv_pc;
    const Real _C2       = 3.0e3/conv_pc;
    const Real _a3       = 12e3/conv_pc;
    const Real _C3       = 0.325; // no units!!
    const Real _vc2      = SQR(2.25e2/conv_kmsc);
    const Real _Tco1[3]  = {log10(1e4)-log10(1e2),5e1/conv_pc,2.5e1/conv_pc};
    const Real _Tco2[3]  = {log10(2e6)-log10(1e4),5e2/conv_pc,5e2/conv_pc};

    // These are the hydrostatic profile arrays, defined over half the z-extent
    AthenaArray<Real> _dens;
    AthenaArray<Real> _temp;
    AthenaArray<Real> _zhaf;

    Real tanhcomp(const Real c[], const Real z) {return 0.5*c[0]*(1+tanh((z-c[1])/c[2]));};
    Real coshcomp(const Real c[], const Real z) {return 0.5*c[0]/SQR(cosh((z-c[1])/c[2]))/c[2];};
    Real phi(const Real z);
    Real dphidz(const Real z);
    Real Tofz(const Real z, const Real n);
    Real dTofzdz(const Real z, const Real n);
};

// Constructor. Initializes the potential variables and calculates hydrostat 
// equilibrium to be used for initialization and boundary conditions.
Potential::Potential(const int iprob, const int itemp, const Real T0, const Real n0, 
                     const Real vx0, const Real rhos0, const Real ramp, const Real x1len,
                     const Real zmin, const Real zmax, const Real awarp, const int kwarp,
                     const int nx2, const Real gam) {
  if (itemp > 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in spiralarm.cpp ProblemGenerator" << std::endl
         << "   class Potential: invalid itemp " << itemp << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  // initialization parameters
  _iprob = iprob;
  _itemp = itemp;
  _T0    = T0; // unless itemp == 2
  _n0    = n0;
  _P0    = n0*T0;
  _vx0   = vx0;
  _ramp  = ramp;
  _x1len = x1len;
  _x2len = zmax-zmin;
  _zmin  = 0.0;
  _zmax  = zmax;
  _awarp = awarp;
  _kwarp = kwarp;
  _nx2   = nx2;
  _gam   = gam;
  _rgc   = 0.5*x1len/PI; // specify the circumference and assume periodic boundaries
  _rhos0 = rhos0;

  _zhaf.NewAthenaArray(_nx2);
  _dens.NewAthenaArray(_nx2); // _nx2 is of global length nz/2+1
  _temp.NewAthenaArray(_nx2);

  _dx2 = _zmax/((Real) (_nx2-1));
  for (int k=0; k<_nx2; k++) 
    _zhaf(k) = ((Real)k) * _dx2;

  return;
};

// To be used for enrolling as static gravitational potential
Real Potential::gravpot(const Real x1, const Real x2, const Real x3, const Real time) {
  // spiral arm contribution (Cox & Gomez 2002)
  Real gamma = ((Real)_narm) * x1/_rgc; // spiral arm in domain center if -L,L
  Real xx2   = _awarp*sin(2.0*PI*((Real)_kwarp)*x1/_x1len);
  Real sumc  = 0.0;
  for (int n=0; n<_nord; n++) {
    int n1  = n+1;
    Real Kn = ((Real)(n1*_narm)) / (_rgc*_sa);
    Real bn = Kn*_H0*(1.0+0.4*Kn*_H0);
    Real Dn = (1.0+Kn*_H0+0.3*SQR(Kn*_H0))/(1.0+0.3*Kn*_H0);
    sumc   += (_Cn[n]/(Kn*Dn))*cos(n1*gamma)/pow(cosh(Kn*xx2/bn),bn);
  }
  Real phis = -4.0*_H0*_rhos0*exp(-(_rgc-_r0)/_rs)*sumc;
  Real x22  = SQR(xx2);
  // disk/halo contribution (Wolfire et al. 1995)
  Real rgc2 = SQR(_rgc);
  Real phi1 = -_C1*_vc2/sqrt(rgc2+SQR(_a1+sqrt(x22+SQR(_b1))));
  Real phi2 = -_C2*_vc2/(_a2+sqrt(x22+rgc2));
  Real phi3 = -_C3*_vc2*log(SQR(_a3)+rgc2+x22);
  Real gfrac = 1.0;//std::min(_ramp*time*_vx0/_x1len,1.0);

  return phi1+phi2+phi3+gfrac*phis;
};

// to be used for initial conditions: hydrostat equilib
Real Potential::phi(const Real z) {
  return gravpot(0.0,z,0.0,0.0);
};

// potential gradient for initial conditions
Real Potential::dphidz(const Real z) {
  Real z2   = SQR(z);
  Real rgc2 = SQR(_rgc);
  Real b12  = SQR(_b1);
  Real dp1dz = -_C1*_vc2*z*(1+_a1/sqrt(z2+b12))/pow(sqrt(rgc2+SQR(_a1+sqrt(z2+b12))),3);
  Real dp2dz = -_C2*_vc2*z/(SQR(_a2+sqrt(z2+rgc2))*sqrt(z2+rgc2));
  Real dp3dz = -2.0*_C3*_vc2*z/(SQR(_a3)+rgc2+z2);
  //std::cout << "[dphidz()]: "
  //          << " z="     << std::scientific << std::setprecision(5) << z
  //          << " dp1dz=" << std::scientific << std::setprecision(5) << dp1dz
  //          << " dp2dz=" << std::scientific << std::setprecision(5) << dp2dz
  //          << " dp3dz=" << std::scientific << std::setprecision(5) << dp3dz
  //          << std::endl;
  return dp1dz + dp2dz + dp3dz;
};

Real Potential::Tofz(const Real z, const Real n) {
  if (_itemp == 0) { 
    return _T0*pow(n/_n0,_gam-1.0);
  } else if (_itemp == 1) {
    Real comps = tanhcomp(_Tco1,z) + tanhcomp(_Tco2,z);
    return pow(10.0,2+comps); 
  } else 
    return sqrt(-1);
};

void Potential::temperature() {
  for (int k=0; k<_nx2; k++) 
    _temp(k) = Tofz(_zhaf(k),_dens(k));
  return;
};

Real Potential::dTofzdz(const Real z, const Real n) {
  if (_itemp == 1) {
    Real ccomps = coshcomp(_Tco1,z) + coshcomp(_Tco2,z);
    Real tcomps = tanhcomp(_Tco1,z) + tanhcomp(_Tco2,z);
    return log(10.0)*ccomps*pow(10.0,tcomps);
  } else 
    return sqrt(-1); 
};

// derivative for hydrostat equilibrium
Real Potential::dydx(const Real z, const Real n) {
  if (_itemp == 0) {
    return _n0*dphidz(z)*pow(n/_n0,2-_gam)/(_gam*_T0);
  } else if (_itemp == 1) {
    return (n/Tofz(z,n)) * (dphidz(z)+dTofzdz(z,n));
  } else if (_itemp == 2) 
    return 0.0;
  
};

// equilibrium temperature
Real Potential::Teq(const Real n) {
  return 0.0;
};

//====================================================================================
// End class Potential
//====================================================================================
//====================================================================================
// Begin class OdeIntegrator. Not fully general, but can just switch out potential,
// which has some standard definitions.
//====================================================================================
class OdeIntegrator {
  public:
    OdeIntegrator(Potential *par, const Real eps);
    ~OdeIntegrator();
    void odeint();

  private:
    Potential* _par;
    int  _nx, _maxit;
    Real _dx, _x0, _x1, _y0, _eps, _pshrink, _pgrow, _safe, _errcon;
    AthenaArray<Real> _a, _b, _c, _dc;
    Real rk45(const Real x0, const Real y0, const Real dx);
    Real rk45single(const Real x0, const Real y0, const Real dx, Real *yerr);
};

OdeIntegrator::OdeIntegrator(Potential* par, const Real eps) {
  _par    = par;
  _x0     = _par->xmin();
  _x1     = _par->xmax();
  _nx     = _par->nx();
  _dx     = (_x1-_x0) / ((Real) _nx-1);
  _y0     = _par->y0();
  _eps    = eps;
  std::cout << "OdeIntegrator: " << _x0 << " " << _x1 << " " << _nx << " " << _dx << " " << _eps << std::endl;
  // parameters for rk45 stepsize control
  _pshrink= -0.25;
  _pgrow  = -0.2;
  _safe   = 0.9;
  _errcon = 1.89e-4;
  _maxit  = 100000 ;
  // butcher tableau
  const Real __a[6]  = {0.0,0.2,0.3,0.6,1.0,0.875};
  const Real __c[6]  = {37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0};
  const Real __d[6]  = {2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25};
  const Real __b1[6] = {0.0,0.2,0.075,0.3,-11.0/54.0,1631.0/55296.0};
  const Real __b2[6] = {0.0,0.0,0.225,-0.9,2.5,175.0/512.0};
  const Real __b3[6] = {0.0,0.0,0.0,1.2,-70.0/27.0,575.0/13824.0};
  const Real __b4[6] = {0.0,0.0,0.0,0.0,35.0/27.0,44275.0/110592.0};
  const Real __b5[6] = {0.0,0.0,0.0,0.0,0.0,253.0/4096.0};
  _a.NewAthenaArray(6);
  _b.NewAthenaArray(5,6); // matrix convention (j,i): i is row, j is column. Note that Athena++ reverses indices.
  _c.NewAthenaArray(6);
  _dc.NewAthenaArray(6);
  for (int i=0; i<6; i++) {
    _a(i)  = __a[i];
    _c(i)  = __c[i];
    _dc(i) = __c[i]-__d[i];
    _b(0,i)= __b1[i];
    _b(1,i)= __b2[i];
    _b(2,i)= __b3[i];
    _b(3,i)= __b4[i];
    _b(4,i)= __b5[i];
  }
  return;
};

OdeIntegrator::~OdeIntegrator() {
  _a.DeleteAthenaArray();
  _b.DeleteAthenaArray();
  _c.DeleteAthenaArray();
  _dc.DeleteAthenaArray();
  return;
};

void OdeIntegrator::odeint() {
  _par->y(0) = _y0;
  for (int k=1; k<_nx; k++) {
    _par->y(k) = rk45(_dx*((Real)(k-1)),_par->y(k-1),_dx);
    std::cout << "[OdeIntegrator::odeint()]: k=" << std::setw(5) << k
              << " x="   << std::scientific << std::setprecision(5) << _dx*((Real)k)
              << " y="   << std::scientific << std::setprecision(5) << _par->y(k)
              << std::endl;
  }
  return;
};

// single step for rk45, returning updated y and error. dx is the current (trial) step.
Real OdeIntegrator::rk45single(const Real x0, const Real y0, const Real dx, Real *yerr) {
  Real dy;                        // updates [arguments in f(x,y)]
  Real dydx[6];                   // derivatives (these are the k1,k2,k3,k4,k5,k6)
  Real yout=y0;                      // result
  *yerr = 0.0;
  dydx[0] = dx*_par->dydx(x0,y0); // first guess
  for (int i=1; i<6; i++) {       // outer loop over k_i
    dy      = 0.0;
    for (int j=0; j<i; j++)       // inner loop over y as argument to f(x,y)
      dy += _b(j,i)*dydx[j];
    dydx[i] = dx*_par->dydx(x0+_a(i)*dx,y0+dy); 
  }
  for (int i=0; i<6; i++) {       // add up the k_i times their weighting factors
    yout  += _c(i) *dydx[i];      // solution
    *yerr += _dc(i)*dydx[i];      // error
  }
  return yout;
};

// driver for rk45, attempting a single step dx. 
Real OdeIntegrator::rk45(const Real x0, const Real y0, const Real dx) {
  Real xt      = 0.0;                                          // temporary independent variable: will count up to dx.
  Real x1      = x0;                                           // keeps track of absolute x position
  int  it      = 0;                                            // iteration counter, as safeguard
  Real dydx    = _par->dydx(x0,y0);                            // first guess
  Real y1      = y0;
  Real y2      = y0;
  Real dxtry   = dx;                                           // starting guess for step size    
  Real dxtmp   = dx;
  int idone    = 0;
  Real yscal, yerr, errmax, xnew, dxnext;
  while ((xt < dx) && (it < _maxit)) {
    yscal = fabs(y1) + fabs(dxtry*dydx);                       // error scaling on last timestep, see NR92, sec 16.2
    idone = 0;                                                 // reset idone
    while (!idone) {                                           // figure out an acceptable stepsize
      y2      = rk45single(x1,y1,dxtry,&yerr);
      errmax  = fabs(yerr/yscal)/_eps;
      if (errmax > 1.0) {                                      // stepsize too large - reduce
        dxtmp = dxtry*_safe*pow(errmax,_pshrink);
        dxtry = std::max(dxtmp,0.1*dxtry);                          // warning! This is only for dxtry > 0.
        xnew  = x1 + dxtry;
        if (xnew == xt) {
          std::stringstream msg;
          msg << "### FATAL ERROR in spiralarm.cpp: OdeIntegrator::rk45" << std::endl
              << "    integration step too small." << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      } else                                                   // stepsize ok - we're done with the trial loop
        idone = 1;
    }
    y1  = y2;                                                  // update so that integration is advanced at next iteration.
    it += 1;
    if (errmax > _errcon) {                                   // if the error is larger than safety, reduce growth rate
      dxnext = _safe*dxtry*pow(errmax,_pgrow);
    } else                                                     // if error less than safety, increase by factor of 5.
      dxnext = 5.0*dxtry;
    
    xt    += dxtry;
    x1    += dxtry;
    dxtry = std::min(dx-xt,dxnext);                                 // guess next timestep - make sure it's flush with dx.
  }
  return y2;
};

//====================================================================================
// End class OdeIntegrator
//====================================================================================
//====================================================================================
// Global variables for boundaries and gravity
AthenaArray<Real> dprofloc, tprofloc;
Real gm1;
Potential *pot;

//========================================================================================
// ... and the wrapper for the gravitational potential
Real gravpot_spirarm(const Real x1, const Real x2, const Real x3, const Real time) {
  return pot->gravpot(x1,x2,x3,time);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spiral arm problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  OdeIntegrator *odeint;
  Real n0, T0, bx0;
  AthenaArray<Real> dprofcmp, tprofcmp;
  int  nx1      = pin->GetInteger("mesh","nx1");
  int  nx2      = pin->GetInteger("mesh","nx2");
  int  nx3      = pin->GetInteger("mesh","nx3");
  int  nz, nzloc, iz, izloc, izs, IZ, IY;
  Real x1min    = pin->GetReal("mesh","x1min");
  Real x1max    = pin->GetReal("mesh","x1max");
  Real x2min    = pin->GetReal("mesh","x2min");
  Real x2max    = pin->GetReal("mesh","x2max");
  Real x3min    = pin->GetReal("mesh","x3min");
  Real x3max    = pin->GetReal("mesh","x3max");
  Real x1len    = x1max-x1min;
  Real x2len    = x2max-x2min;
  Real x3len    = x3max-x3min;
  Real zmin,zmax,zloc0;
  Real gamma    = peos->GetGamma();
  gm1           = gamma-1.0;

  if (nx3 > 1) {
    izs   = ks;
    nz    = nx3;
    nzloc = ke-ks+1;
    zmin  = x3min;
    zmax  = x3max;
    zloc0 = pcoord->x3v(izs);
    IY    = IM2;
    IZ    = IM3;
  } else {
    izs   = js;
    nz    = nx2;
    nzloc = je-js+1;
    zmin  = x2min;
    zmax  = x2max;
    zloc0 = pcoord->x2v(izs);
    IY    = IM3;
    IZ    = IM2;
  }
  Real zlen     = zmax-zmin;

  int iprob     = pin->GetInteger("problem","iprob");
  int itemp     = pin->GetInteger("problem","itemp"); // 0: constant, 1: tanh(z), 2: thermal equilibrium
  if (itemp < 2) 
    T0          = pin->GetReal("problem","T0"); // Temperature at midplane. For itemp==1, 0.01*Thalo
  n0            = pin->GetReal("problem","n0"); // midplane density
  Real ramp     = pin->GetReal("problem","ramp"); // ramping time in crossing times for spiral arm contribution to potential 
  Real vx0      = pin->GetReal("problem","vx0"); // velocity in ISM units (somewhere around 200)
  Real rhos0    = pin->GetReal("problem","rhos0"); // stellar mass density (in ISM units) Somehwere around 1...10
  Real awarp    = pin->GetReal("problem","awarp"); // amplitude (in pc) of periodic warp 
  Real kwarp    = pin->GetInteger("problem","kwarp"); // wavenumber of periodic warp

  pot           = new Potential(iprob, itemp, T0, n0, vx0, rhos0, ramp, x1len, zmin, zmax, awarp, kwarp, nz/2+1,gamma);
  odeint        = new OdeIntegrator(pot, 1e-7);
  odeint->odeint();
  pot->temperature();
  // at this point, we have the density half-profile, i.e. we need to calculate the rest. 
  dprofcmp.NewAthenaArray(nz+1);
  tprofcmp.NewAthenaArray(nz+1);
  dprofloc.NewAthenaArray(nzloc);
  tprofloc.NewAthenaArray(nzloc);
  for (iz=0; iz<=nz/2; iz++) {
    dprofcmp(nz/2+iz) = pot->dens(iz);
    dprofcmp(nz/2-iz) = pot->dens(iz);
    tprofcmp(nz/2+iz) = pot->temp(iz);
    tprofcmp(nz/2-iz) = pot->temp(iz);
  } 
  izloc = (int) (zloc0-zmin)/(zmax-zmin)*((Real) nz); // starting position
  for (iz=0; iz<nzloc; iz++) {
    dprofloc(iz) = 0.5*(dprofcmp(iz+izloc)+dprofcmp(iz+izloc+1));
    tprofloc(iz) = 0.5*(tprofcmp(iz+izloc)+tprofcmp(iz+izloc+1));
    std::cout << "[Problem]: iz=" << std::setw(5) <<iz 
              << " dprofloc=" << std::scientific << std::setprecision(5) << dprofloc(iz)
              << " tprofloc=" << std::scientific << std::setprecision(5) << tprofloc(iz)
              << std::endl;

  }
  // now fill the main grid
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
      if (nx3 > 1) {
          iz = k;
        } else {
          iz = j;
        }
        phydro->u(IDN,k,j,i) = dprofloc(iz-izs);
        phydro->u(IM1,k,j,i) = dprofloc(iz-izs)*vx0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) =  tprofloc(iz-izs)*dprofloc(iz-izs)/gm1 
                                + 0.5*( SQR(phydro->u(IM1,k,j,i))
                                       +SQR(phydro->u(IM2,k,j,i))
                                       +SQR(phydro->u(IM3,k,j,i)))
                                     /dprofloc(iz-izs);
        }
        if (DUAL_ENERGY) {
          phydro->u(IIE,k,j,i) = tprofloc(iz-izs)*dprofloc(iz-izs)/gm1;
        }
      }
    }
  }

  // initialize interface B and total energy
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (nx3 > 1) {
            iz = k;
            pfield->b.x2f(k,j,i) = 0.0;//by0*sqrt(csound*csound*n00*beta0)*pow(dprofloc(iz-izs)/n00,bpow);
          } else {
            iz = j;
            pfield->b.x2f(k,j,i) = 0.0;//by0;
          }
        }
      }
    }
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (nx3 > 1) {
            iz = k;
            pfield->b.x3f(k,j,i) = 0.0;//bz0;
          } else {
            iz = j;
            pfield->b.x3f(k,j,i) = 0.0;//bz0*sqrt(csound*csound*n00*beta0)*pow(dprofloc(iz-izs)/n00,bpow);
          }
        }
      }
    }
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          phydro->u(IEN,k,j,i) += 0.5*(SQR(pfield->b.x1f(k,j,i))+SQR(pfield->b.x2f(k,j,i))+SQR(pfield->b.x3f(k,j,i)));
        }
      }
    }
  }
  dprofcmp.DeleteAthenaArray();
  tprofcmp.DeleteAthenaArray();

  // set the mean density
  Real dmean = GetMeanDensity(pcoord, phydro);
  pmy_mesh->SetMeanDensity(dmean);

  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop(void)
//  \brief User-defined actions. No source terms!!
//========================================================================================
//

void MeshBlock::UserWorkInLoop(void) {
  Real d0 = GetMeanDensity(pcoord, phydro);
  pmy_mesh->SetMeanDensity(d0);
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Check radius of sphere to make sure it is round
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}

//========================================================================================
//! \fn void SpiralInnerX1(...)
//  \brief Inner (inflow) X1 boundary conditions for spiral arm
//========================================================================================

void SpiralInflowInnerX1(MeshBlock *pmb, Coordinates *pco,
    AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js,
    int je, int ks, int ke, int ngh) {

  int iz,izs,IY,IZ;
  int nx3 = ke-ks+1;
  if (nx3 > 1) {
    izs = ks;
    IY  = IM2;
    IZ  = IM3;
  } else {
    izs = js;
    IY  = IM3;
    IZ  = IM2;
  }

  // Assign fields 
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++) {
          b.x1f(k,j,i) = 0.0;//bx0_bc;
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=0; i<ngh; i++) {
          if (nx3 > 1) {
            iz = k;
            b.x2f(k,j,i) = 0.0;//by0_bc*sqrt(csound_bc*csound_bc*n00_bc*beta0_bc)*pow(dprofloc(iz-izs)/n00_bc,bpow_bc);
          } else {
            iz = j;
            b.x2f(k,j,i) = 0.0;//by0_bc;
          }
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++) {
          if (nx3 > 1) {
            iz = k;
            b.x3f(k,j,i) = 0.0;//bz0_bc;
          } else {
            iz = j;
            b.x3f(k,j,i) = 0.0;//bz0_bc*sqrt(csound_bc*csound_bc*n00_bc*beta0_bc)*pow(dprofloc(iz-izs)/n00_bc,bpow_bc);
          }
        }
      }
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=0; i<ngh; i++) {
        if (nx3 > 1) {
          iz = k;
        } else {
          iz = j;
        }
        pmb->phydro->u(IDN,k,j,i) = dprofloc(iz-izs);
        pmb->phydro->u(IM1,k,j,i) = dprofloc(iz-izs)*pot->vx0();
        pmb->phydro->u(IY ,k,j,i) = 0.0;
        pmb->phydro->u(IZ ,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          pmb->phydro->u(IEN,k,j,i) =  tprofloc(iz-izs)*dprofloc(iz-izs)/gm1 
                                     + 0.5*( SQR(pmb->phydro->u(IM1,k,j,i))
                                            +SQR(pmb->phydro->u(IM2,k,j,i))
                                            +SQR(pmb->phydro->u(IM3,k,j,i)))
                                          /dprofloc(iz-izs);
        }
        if (DUAL_ENERGY) {
          pmb->phydro->u(IIE,k,j,i) = tprofloc(iz-izs)*dprofloc(iz-izs)/gm1;
        }
        if (MAGNETIC_FIELDS_ENABLED) {
          pmb->phydro->u(IEN,k,j,i) += 0.5*( SQR(b.x1f(k,j,i))
                                            +SQR(b.x2f(k,j,i))
                                            +SQR(b.x3f(k,j,i)));
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void SpiralOutflowOuterX1(...)
//  \brief Outer (outflow) X1 boundary conditions for spiral arm
//========================================================================================

void SpiralOutflowOuterX1(MeshBlock *pmb, Coordinates *pco,
    AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js,
    int je, int ks, int ke, int ngh) {

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=1; i<=ngh; i++) {
          b.x1f(k,j,ie+i) = b.x1f(k,j,ie);
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=1; i<=ngh; i++) {
          b.x2f(k,j,ie+i) = b.x2f(k,j,ie);
        }
      }
    }     
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=1; i<=ngh; i++) {
          b.x3f(k,j,ie+i) = b.x3f(k,j,ie);
        }        
      }
    }
  }
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=1; i<=ngh; i++) {
      }
    }
  }
  return;
}


