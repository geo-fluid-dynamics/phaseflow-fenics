# phaseflow-fenics
Phaseflow simulates the unsteady conservation of mass, momentum, and energy for an incompressible fluid using the Python finite element library FEniCS.

Currently only a homogeneous fluid is supported. The project is currently under heavy development to support phase-change materials, particularly the melting and freezing of water-ice.

Author: Alexander G. Zimmerman <alex.g.zimmerman@gmail.com>

[![Build Status](https://travis-ci.org/alexanderzimmerman/phaseflow-fenics.svg?branch=master)](https://travis-ci.org/alexanderzimmerman/phaseflow-fenics) (<b>Continuous integration status</b>; click the button to go to Travis-CI)

## Current capabilities
- Incompressible Navier-Stokes (steady and unsteady), lid-driven cavity benchmark
- Heat conduction
- Not yet successfully verified: Natural convection, where the momentum equation includes temperature-based bouyancy forces per the Boussinesq approximation
