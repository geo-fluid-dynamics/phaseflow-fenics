"""**pure_with_constant_properties.py** contains code specific to pure materials with constant properties.

For example, this is useful for an octadecane phase-change material.
"""
import fenics
import phaseflow
import phaseflow.pure


class Model(phaseflow.core.Model):

    def __init__(self,
            mesh,
            boundary_conditions = None, 
            buoyancy = None,
            semi_phasefield_mapping = None,
            timestep_bounds = (1.e-4, 1., 1.e12),
            quadrature_degree = None,
            prandtl_number = 1.,
            stefan_number = 1.,
            liquid_viscosity = 1.,
            solid_viscosity = 1.e8,
            penalty_parameter = 1.e-7,
            automatic_jacobian = False):
        """
        Parameters
        ----------
        state : phaseflow.State
        
        timestep_size : float
        
        boundary_conditions : [fenics.DirichletBoundaryCondition,]
        
        buoyancy : phaseflow.ContinuousFunctionOfTemperature
        
        semi_phasefield_mapping : phaseflow.ContinuousFunctionOfTemperature
        """
        if semi_phasefield_mapping is None:
        
            semi_phasefield_mapping = phaseflow.pure.ConstantFunction(0.)
            
        if buoyancy is None:
        
            buoyancy = phaseflow.pure.ConstantFunction((0.,)*mesh.type().dim())
        
        
        self.semi_phasefield_mapping = semi_phasefield_mapping
        
        self.buoyancy = buoyancy
        
        self.prandtl_number = prandtl_number
        
        self.stefan_number = stefan_number
        
        self.liquid_viscosity = liquid_viscosity
        
        self.solid_viscosity = solid_viscosity
        
        self.penalty_parameter = penalty_parameter
        
        self.automatic_jacobian = automatic_jacobian
        
        phaseflow.core.Model.__init__(self,
            mesh = mesh,
            element = phaseflow.pure.make_mixed_element(mesh.ufl_cell()),
            boundary_conditions = boundary_conditions,
            timestep_bounds = timestep_bounds,
            quadrature_degree = quadrature_degree)
        
        
    def setup_variational_form(self):
        """ Define the nonlinear variational form. """
        
        
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
        
        
        Delta_t = self.Delta_t
        
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
        
        W = self.function_space
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(W)
        
        w = self.state.solution
        
        p, u, T = fenics.split(w)
        
        w_n = self.old_state.solution
         
        p_n, u_n, T_n = fenics.split(w_n)
        
        dx = self.integration_metric
    
        self.variational_form = (
            b(u, psi_p) - psi_p*gamma*p
            + dot(psi_u, 1./Delta_t*(u - u_n) + f_B(T))
            + c(u, u, psi_u) + b(psi_u, p) + a(mu(phi,T), u, psi_u)
            + 1./Delta_t*psi_T*(T - T_n - 1./Ste*(phi(T) - phi(T_n)))
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)        
            )*dx

        
        # Set the derivative of the variational form for linearizing the nonlinear problem.
        if self.automatic_jacobian:
        
            self.derivative_of_variational_form = None
            
        else:  # Set the manually derived Gateaux derivative in variational form.
        
            delta_w = fenics.TrialFunction(W)
            
            df_B = self.buoyancy.derivative_function
            
            dphi = self.semi_phasefield_mapping.derivative_function
            
            dmu = phase_dependent_viscosity.derivative_function
            
            delta_p, delta_u, delta_T = fenics.split(delta_w)
            
            w_k = w
            
            p_k, u_k, T_k = fenics.split(w_k)
            
            self.derivative_of_variational_form = (
                b(delta_u, psi_p) - psi_p*gamma*delta_p 
                + dot(psi_u, 1./Delta_t*delta_u + delta_T*df_B(T_k))
                + c(u_k, delta_u, psi_u) + c(delta_u, u_k, psi_u) 
                + b(psi_u, delta_p) 
                + a(delta_T*dmu(phi, T_k), u_k, psi_u) + a(mu(phi, T_k), delta_u, psi_u) 
                + 1./Delta_t*psi_T*delta_T*(1. - 1./Ste*dphi(T_k))
                + dot(grad(psi_T), 1./Pr*grad(delta_T) - T_k*delta_u - delta_T*u_k)
                )*dx
                
                