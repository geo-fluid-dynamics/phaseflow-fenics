""" **test_abstract_phasechange_simulation.py** tests **phaseflow/abstract_phasechange_simulation.py**,
Phaseflow's module for the natural and compositional convection of binary alloys."""
import phaseflow
import fenics

    
class AbstractMMSVerificationSimulation(
        phaseflow.abstract_phasechange_simulation.AbstractPhaseChangeSimulation):

    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  0}),
            setup_solver = True):
        
        class Boundaries(fenics.SubDomain):

            def inside(self, x, on_boundary):

                return on_boundary
        
        self.boundaries = Boundaries()
        
        self.time = fenics.Variable(0., name = "t")
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver)
    
    def source_terms(self):
    
        p_M, u_M, T_M, C_M = self.manufactured_solution()
        
        gamma = self.pressure_penalty_factor
        
        b = self.buoyancy(T = T_M, C = C_M)
        
        phi = self.semi_phasefield(T = T_M, C = C_M)
        
        mu_L = self.liquid_viscosity
        
        mu_S = self.solid_viscosity
        
        mu = mu_L + (mu_S - mu_L)*phi
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        Sc = self.schmidt_number
        
        t = self.time
        
        div, diff, inner = fenics.div, fenics.diff, fenics.inner
        
        s_p = div(u_M) - gamma*p_M
        
        s_u = diff(u_M, t) + inner(u_M, grad(u_M)) + grad(p_M) - \
            2.*div(mu*inner(sym(grad(u_M)), sym(grad(u_M)))) + b
            
        s_T = diff(T_M, t) + inner(u_M, grad(T_M)) + 1./Pr*div(grad(T_M)) - 1./Ste*diff(phi, t)
        
        s_C = (1. - phi)*diff(C_M, t) + inner(u_M, grad(C_M)) - 1./Sc*div((1. - phi)*grad(C_M)) - \
            C_M*diff(phi, t)
    
        return s_p, s_u, s_T, s_C
    
    def governing_form(self):
    
        F = super().governing_form()
        
        s_p, s_u, s_T, s_C = self.source_terms()
    
        psi_p, psi_u, psi_T, psi_C = fenics.TestFunctions(self.function_space)
        
        F_M = F - psi_p*s_p - fenics.inner(psi_u, s_u) - psi_T*s_T - psi_C*s_C
        
        return F_M
    
    def initial_values(self):
        
        p_M, u_M, T_M, C_M = self.manufactured_solution()
        
        w0 = fenics.interpolate(
            fenics.Expression(
                (p_M, u_M[0], u_M[1], T_M, C_M),
                t = 0.,
                element = self.element()),
            self.function_space)
        
        return w0
            
    def boundary_conditions(self):
        
        p_M, u_M, T_M, C_M = self.manufactured_solution()
        
        return [
            fenics.DirichletBC(
                self.function_space, 
                (p_M, u_M[0], u_M[1], T_M, C_M), 
                self.boundaries),]
            
class MeltingAndRefreezingMMSVerificationSimulation(AbstractMMSVerificationSimulation):

    def __init__(self, 
            uniform_gridsize = 8):
        
        self.uniform_gridsize = uniform_gridsize
        
        self.endtime = fenics.Constant(1., name = "t_f")
        
        super().__init__()
        
    def coarse_mesh(self):
        
        M = self.uniform_gridsize
        
        return fenics.UnitSquareMesh(M, M)
        
    def initial_mesh(self):
        
        return self.coarse_mesh()
        
    def manufactured_solution(self):
    
        sin, cos, pi = fenics.sin, fenics.cos, fenics.pi
        
        t = self.time
        
        t_f = self.endtime
        
        x, y = fenics.SpatialCoordinate(self.mesh)
        
        C_M = (t/t_f)**2*sin(x)*sin(2.*y)
        
        T_M = cos(pi*(2.*t/t_f) - 1.)*(1. - sin(x)*sin(2.*y))
        
        phi = self.semi_phasefield(T = T_M, C = C_M)
        
        u_M = (1. - phi)*(t/t_f)**2*sin(x)*sin(2.*y)
        
        p_M = -0.5*u_M**2
    
        return p_M, u_M, T_M, C_M
    
    
def test_mms_verify_melting_and_refreezing():

    timestep_sizes = (1./8.,)
    
    uniform_grid_sizes = (8,)
    
    for Delta_t in timestep_sizes:
    
        for M in uniform_grid_sizes:
        
            sim = MeltingAndRefreezingMMSVerificationSimulation(uniform_gridsize = M)
    
            sim.timestep_size.assign(Delta_t)
            
            sim.assign_initial_values()
            
            t_f = sim.endtime.__float__()
            
            t = 0.
            
            while t < t_f:
            
                sim.solve_with_auto_regularization()
    
                sim.advance()
                
                t = sim.time.__float__()
