import phaseflow
import fenics


class CavityMeltingSimulation(
        phaseflow.abstract_heated_cavity_phasechange_simulation.AbstractHeatedCavityPhaseChangeSimulation):

    def __init__(self, 
            initial_uniform_gridsize = 1,
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            setup_solver = True):
        
        self.initial_hot_wall_refinement_cycles = 6
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure, 
            setup_solver = setup_solver,
            initial_uniform_gridsize = initial_uniform_gridsize)
        
        self.hot_wall_temperature.assign(1.)
    
        self.cold_wall_temperature.assign(-0.11)
        
        self.initial_concentration.assign(1.)
        
        self.temperature_rayleigh_number.assign(3.27e5)
        
        self.prandtl_number.assign(56.2)
        
        Ra_T = self.temperature_rayleigh_number.__float__()
        
        Pr = self.prandtl_number.__float__()
        
        self.concentration_rayleigh_number.assign(-0.3*Ra_T/Pr)
        
        self.stefan_number.assign(0.045)
        
        Le = 100.
        
        self.schmidt_number.assign(Pr*Le)
        
        self.pure_liquidus_temperature.assign(0.)
        
        self.liquidus_slope.assign(-0.1)
        
        self.regularization_central_temperature_offset.assign(0.01)
        
        self.regularization_smoothing_parameter.assign(0.025)
    
    def initial_mesh(self):
    
        mesh = self.coarse_mesh()
        
        for cycle in range(self.initial_hot_wall_refinement_cycles):
            
            edge_markers = fenics.MeshFunction("bool", mesh, 1, False)
            
            self.hot_wall.mark(edge_markers, True)
            
            fenics.adapt(mesh, edge_markers)
            
            mesh = mesh.child()
            
        return mesh
    
    def initial_values(self):
    
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", 
                 "0.", 
                 "0.", 
                 "(T_h - T_c)*(x[0] < x_m0) + T_c", 
                 "C_0*(x[0] < x_m0)",
                 "1. - (x[0] < x_m0)", ),
                T_h = self.hot_wall_temperature, 
                T_c = self.cold_wall_temperature,
                C_0 = self.initial_concentration,
                x_m0 = 1./2.**(self.initial_hot_wall_refinement_cycles - 1),
                element = self.element()),
            self.function_space)
            
        return initial_values
        
    def deepcopy(self):
    
        sim = super().deepcopy()
        
        sim.initial_hot_wall_refinement_cycles = 0 + self.initial_hot_wall_refinement_cycles
        
        return sim
        