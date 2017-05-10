# phase-flow
PhaseFlow simulates the unsteady conservation of mass, momentum, and energy for an incompressible fluid using the Python finite element library FEniCS.

Currently only a homogeneous fluid is supported. The project is currently under heavy development to support phase-change materials, particularly the melting and freezing of water-ice.

Author: Alexander G. Zimmerman zimmerman@aices.rwth-aachen.de

## Current capabilities
- Incompressible Navier-Stokes (steady and unsteady), lid-driven cavity benchmark
- Heat conduction
- Not yet successfully verified: Natural convection, where the momentum equation includes temperature-based bouyancy forces per the Boussinesq approximation
