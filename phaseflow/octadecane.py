""" **octadecane.py** implements the convection-coupled melting of octadecane phase-change materials. """
import fenics
import phaseflow

    
class Simulation(phaseflow.simulation.Simulation):

    def __init__(self):
        """ This extends `Simulation.__init__` with attributes needed for the octadecane model. """
        phaseflow.simulation.Simulation.__init__(self)
        
        self.timestep_size = 1.
        
        self.rayleigh_number = 1.
        
        self.prandtl_number = 1.
        
        self.stefan_number = 1.
        
        self.gravity = (0., -1.)
        
        self.liquid_viscosity = 1.
        
        self.solid_viscosity = 1.e8
        
        self.penalty_parameter = 1.e-7
        
        self.regularization_central_temperature = 0.
        
        self.regularization_smoothing_parameter = 0.01
        
        self.pressure_element_degree = 1
        
        self.temperature_element_degree = 1
        
        
    def update_element(self):
        """ Set the mixed element from @cite{danaila2014newton}. """
        pressure_element = fenics.FiniteElement("P", self.mesh.ufl_cell(), self.pressure_element_degree)
        
        velocity_element_degree = self.pressure_element_degree + 1
        
        velocity_element = fenics.VectorElement("P", self.mesh.ufl_cell(), velocity_element_degree)

        temperature_element = fenics.FiniteElement(
            "P", self.mesh.ufl_cell(), self.temperature_element_degree)
        
        self.element = fenics.MixedElement([pressure_element, velocity_element, temperature_element])
        
        
    def update_governing_form(self):
        """ Implement the variational form from @cite{zimmerman2018monolithic}. """
        Delta_t = fenics.Constant(self.timestep_size)
        
        Pr = fenics.Constant(self.prandtl_number)
        
        Ra = fenics.Constant(self.rayleigh_number)
        
        Ste = fenics.Constant(self.stefan_number)
        
        g = fenics.Constant(self.gravity)
        
        T_r = fenics.Constant(self.regularization_central_temperature)
        
        r = fenics.Constant(self.regularization_smoothing_parameter)
        
        mu_L = fenics.Constant(self.liquid_viscosity)
        
        mu_S = fenics.Constant(self.solid_viscosity)
        
        gamma = fenics.Constant(self.penalty_parameter)
        
        def f_B(T):
            """ Idealized linear Boussinesq Buoyancy with $Re = 1$ """
            return T*Ra*g/Pr
        
        
        def phi(T):
            """ Semi-phase-field mapping from temperature """
            return 0.5*(1. + fenics.tanh((T_r - T)/r))
            
            
        def mu(phi_of_T):
            """ Phase dependent viscosity """
            return mu_L + (mu_S - mu_L)*phi_of_T
        
        
        p, u, T = fenics.split(self.state.solution)
         
        p_n, u_n, T_n = fenics.split(self.old_state.solution)
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(self.state.solution.function_space())
        
        inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
        
        self.governing_form = (
            -div(u)*psi_p 
            - psi_p*gamma*p
            + dot(psi_u, 1./Delta_t*(u - u_n) + f_B(T))
            + dot(dot(grad(u), u), psi_u) 
            - div(psi_u)*p 
            + 2.*mu(phi(T))*inner(sym(grad(u)), sym(grad(psi_u)))
            + 1./Delta_t*psi_T*(T - T_n - 1./Ste*(phi(T) - phi(T_n)))
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
            )*self.integration_metric
            
        self.semi_phasefield_mapping = phi  # This must be shared for adaptive mesh refinement.
        
        self.fenics_timestep_size = Delta_t  # This must be shared for adaptive time stepping.

        