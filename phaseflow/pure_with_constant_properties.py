"""**pure_with_constant_properties.py** contains code specific to pure materials with constant properties.

For example, this is useful for an octadecane phase-change material.
"""
import fenics
import phaseflow
import phaseflow.pure


class WeakForm(phaseflow.core.WeakForm):

    def __init__(self,
            solutions,
            integration_metric = fenics.dx,
            buoyancy = None,
            semi_phasefield_mapping = None,
            timestep_size = 1.,
            quadrature_degree = None,
            prandtl_number = 1.,
            stefan_number = 1.,
            liquid_viscosity = 1.,
            solid_viscosity = 1.e8,
            penalty_parameter = 1.e-7):

            
        #  Handle arguments.
        if semi_phasefield_mapping is None:
        
            semi_phasefield_mapping = phaseflow.pure.ConstantFunction(0.)
            
        if buoyancy is None:
        
            buoyancy = phaseflow.pure.ConstantFunction((0.,)*mesh.type().dim())
            
        buoyancy = self.buoyancy
        
        Delta_t = fenics.Constant(timestep_size)
        
        self.Delta_t = Delta_t
        
        Pr = fenics.Constant(self.prandtl_number)
        
        Ste = fenics.Constant(self.stefan_number)
        
        f_B = self.buoyancy.function
        
        phi = self.semi_phasefield_mapping
        
        gamma = fenics.Constant(self.penalty_parameter)
        
        mu_L = fenics.Constant(self.liquid_viscosity)
        
        mu_S = fenics.Constant(self.solid_viscosity)
        
        phase_dependent_viscosity = phaseflow.pure.PhaseDependentMaterialProperty(
            liquid_value = mu_L,
            solid_value = mu_S)
            
        mu = phase_dependent_viscosity.function
        
        dx = integration_metric
        
        w = solutions[0]
        
        w_n = solutions[1]
        
        W = w.function_space()
        
        p, u, T = fenics.split(w)
         
        p_n, u_n, T_n = fenics.split(w_n)
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(W)
        
        
        # Set local names for math operators to improve readability.
        inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        
        #The forms a, b, and c follow the common notation from  huerta2003fefluids.
        def b(u, p):
        
            return -div(u)*p  # Divergence
        
        
        def D(u):
        
            return sym(grad(u))  # Symmetric part of velocity gradient
        
        
        def a(mu, u, v):
            
            return 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
        
        
        def c(u, z, v):
            
            return dot(dot(grad(z), u), v)  # Convection of the velocity field
        
        
        self.fenics_variational_form = (
            b(u, psi_p) - psi_p*gamma*p
            + dot(psi_u, 1./Delta_t*(u - u_n) + f_B(T))
            + c(u, u, psi_u) + b(psi_u, p) + a(mu(phi,T), u, psi_u)
            + 1./Delta_t*psi_T*(T - T_n - 1./Ste*(phi(T) - phi(T_n)))
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)        
            )*dx

                
    def set_timestep_size(self, value):
        """ Set the time step size such that the new value will be used in the weak form."""
        self.Delta_t.assign(value)
        