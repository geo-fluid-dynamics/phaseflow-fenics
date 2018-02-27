"""**pure_with_constant_properties.py** contains code specific to pure materials with constant properties.

For example, this is useful for an octadecane phase-change material.
"""
import fenics
import phaseflow

    
class Simulation(phaseflow.simulation.Simulation):

    def __init__(self):

        self.timestep_size = 1.
        
        self.rayleigh_numer = 1.
        
        self.prandtl_number = 1.
        
        self.stefan_number = 1.
        
        self.gravity = (0., -1.)
        
        self.liquid_viscosity = 1.
        
        self.solid_viscosity = 1.e-8
        
        self.penalty_parameter = 1.e-7
        
        self.regularization_central_temperature = 0.
        
        self.regularization_smoothing_parameter = 0.01
        
        self.pressure_element_degree = 1
        
        self.temperature_element_degree = 1
        
        self.semi_phasefield_mapping = None
        
        phaseflow.simulation.Simulation.__init__(self)
        
        
    def update_element(self):
        
        pressure_element = fenics.FiniteElement("P", self.mesh.ufl_cell(), self.pressure_element_degree)
        
        velocity_element_degree = self.pressure_element_degree + 1
        
        velocity_element = fenics.VectorElement("P", self.mesh.ufl_cell(), velocity_element_degree)

        temperature_element = fenics.FiniteElement(
            "P", self.mesh.ufl_cell(), self.temperature_element_degree)
        
        self.element = fenics.MixedElement([pressure_element, velocity_element, temperature_element])
        
        
    def update_governing_form(self):
    
        Delta_t = fenics.Constant(self.timestep_size)
        
        Pr = fenics.Constant(self.prandtl_number)
        
        Ra = fenics.Constant(self.rayleigh_numer)
        
        Ste = fenics.Constant(self.stefan_number)
        
        g = fenics.Constant(self.gravity)
        
        Re = fenics.Constant(1.)
        
        def f_B(T):
        
            return T*Ra/(Pr*Re**2)*g
        
        
        T_r = fenics.Constant(self.regularization_central_temperature)
        
        r = fenics.Constant(self.regularization_smoothing_parameter)
        
        def phi(T):
            """ Semi-phase-field mapping from temperature. """
            return 0.5*(1. + fenics.tanh((T_r - T)/r))
            
            
        self.semi_phasefield_mapping = phi
        
        mu_L = fenics.Constant(self.liquid_viscosity)
        
        mu_S = fenics.Constant(self.solid_viscosity)
            
        def mu(phi_of_T):
        
            return mu_L + (mu_S - mu_L)*phi_of_T
        
        
        gamma = fenics.Constant(self.penalty_parameter)
        
        dx = self.integration_metric
        
        w = self.state.solution
        
        w_n = self.old_state.solution
        
        p, u, T = fenics.split(w)
         
        p_n, u_n, T_n = fenics.split(w_n)
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(w.function_space())
        
        
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
        
        
        self.governing_form = (
            b(u, psi_p) - psi_p*gamma*p
            + dot(psi_u, 1./Delta_t*(u - u_n) + f_B(T))
            + c(u, u, psi_u) + b(psi_u, p) + a(mu(phi(T)), u, psi_u)
            + 1./Delta_t*psi_T*(T - T_n - 1./Ste*(phi(T) - phi(T_n)))
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)        
            )*dx

        