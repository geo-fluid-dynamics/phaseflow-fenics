""" **phasechange_simulation.py** implements the convection-coupled melting of phase-change materials. """
import fenics
import phaseflow

    
class PhaseChangeSimulation(phaseflow.simulation.Simulation):

    def __init__(self):
        """ This extends the `__init__` method with attributes for the convection-coupled phase-change model. """
        phaseflow.simulation.Simulation.__init__(self)
        
        self.timestep_size = 1.
        
        self.rayleigh_number = 1.
        
        self.prandtl_number = 1.
        
        self.stefan_number = 1.
        
        self.gravity = (0., -1.)
        
        self.liquid_viscosity = 1.
        
        self.liquid_thermal_conductivity = 1.
        
        self.liquid_heat_capacity = 1.
        
        self.solid_viscosity = 1.e8
        
        self.solid_thermal_conductivity = 1.
        
        self.solid_heat_capacity = 1.
        
        self.penalty_parameter = 1.e-7
        
        self.regularization_central_temperature = 0.
        
        self.regularization_smoothing_parameter = 0.01
        
        self.temperature_element_degree = 1
        
    
    def setup_derived_attributes(self):
        
        phaseflow.simulation.Simulation.setup_derived_attributes(self)
        
        """ We have to handle the time step size carefully for adaptive time stepping. """
        self.fenics_timestep_size = fenics.Constant(self.timestep_size)
        
        
    def setup_element(self):
        """ Implement the mixed element per @cite{danaila2014newton}. """
        pressure_element = fenics.FiniteElement("P", 
            self.mesh.ufl_cell(), self.temperature_element_degree)
        
        velocity_element = fenics.VectorElement("P", 
            self.mesh.ufl_cell(), self.temperature_element_degree + 1)

        temperature_element = fenics.FiniteElement(
            "P", self.mesh.ufl_cell(), self.temperature_element_degree)
        
        self.element = fenics.MixedElement([pressure_element, velocity_element, temperature_element])
        
        
    def make_buoyancy_function(self):

        Pr = fenics.Constant(self.prandtl_number)
        
        Ra = fenics.Constant(self.rayleigh_number)
        
        g = fenics.Constant(self.gravity)
        
        def f_B(T):
            """ Idealized linear Boussinesq Buoyancy with $Re = 1$ """
            return T*Ra*g/Pr
            
        return f_B
        
    
    def make_semi_phasefield_function(self):
        """ Semi-phase-field mapping from temperature """
        T_r = fenics.Constant(self.regularization_central_temperature)
        
        r = fenics.Constant(self.regularization_smoothing_parameter)
        
        def phi(T):
        
            return 0.5*(1. + fenics.tanh((T_r - T)/r))
    
        return phi
        
        
    def make_phase_dependent_material_property_function(self, P_L, P_S):
        """ Phase dependent material property.

        Parameters
        ----------
        P_L : float
            The value in liquid state.
            
        P_S : float
            The value in solid state.
        """
        def P(phi):
            """ 
            
            Parameters
            ----------
            phi : float
                0. <= phi <= 1.
            """
            return P_L + (P_S - P_L)*phi
            
        return P
        
    
    def setup_governing_form(self):
        """ Implement the variational form per @cite{zimmerman2018monolithic}. """
        Pr = fenics.Constant(self.prandtl_number)
        
        Ste = fenics.Constant(self.stefan_number)
        
        f_B = self.make_buoyancy_function()
        
        phi = self.make_semi_phasefield_function()
        
        mu = self.make_phase_dependent_material_property_function(
            P_L = fenics.Constant(self.liquid_viscosity),
            P_S = fenics.Constant(self.solid_viscosity))
        
        gamma = fenics.Constant(self.penalty_parameter)
        
        Delta_t = self.fenics_timestep_size
        
        p, u, T = fenics.split(self.state.solution)
         
        p_n, u_n, T_n = fenics.split(self.old_state.solution)
        
        psi_p, psi_u, psi_T = fenics.TestFunctions(self.state.solution.function_space())
        
        dx = self.integration_metric
        
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
            )*dx
        