""" **test_heated_cavity_phasechange_simulation.py** tests **phaseflow/heated_cavity_phasechange_simulation.py** """
import phaseflow
import fenics
import tempfile


def test__compositional_convection_coupled_melting_benchmark__amr__regression__ci__():
    
    sim = phaseflow.cavity_melting_simulation.CavityMeltingSimulation()
    
    sim.assign_initial_values()
    
    sim.timestep_size.assign(10.)
    
    for it, epsilon_M in zip(range(4), (0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3)):
    
        if it == 1:
        
            sim.regularization_sequence = None
    
        sim.solve_with_auto_regularization(goal_tolerance = epsilon_M)
        
        sim.advance()
    
    p_fine, u_fine, T_fine, C_fine = fenics.split(sim.solution)
    
    phi = sim.semi_phasefield(T = T_fine, C = C_fine)
    
    expected_solid_area =  0.7405
    
    solid_area = fenics.assemble(phi*fenics.dx)
    
    tolerance = 1.e-4
    
    assert(abs(solid_area - expected_solid_area) < tolerance)

    
def test__write_solution__ci__():

    sim = phaseflow.cavity_melting_simulation.CavityMeltingSimulation()
    
    with phaseflow.helpers.SolutionFile(tempfile.mkdtemp() + "/test__write_solution/solution.xdmf") as solution_file:
    
        sim.write_solution(solution_file)

        
class CavityMeltingSimulationWithoutConcentration(phaseflow.cavity_melting_simulation.CavityMeltingSimulation):
    
    def __init__(self, 
            time_order = 1, 
            integration_measure = fenics.dx(metadata={"quadrature_degree":  8}),
            setup_solver = True):
        
        super().__init__(
            time_order = time_order, 
            integration_measure = integration_measure,
            setup_solver = setup_solver)
        
        
        self.hot_wall_temperature.assign(1.)
        
        self.cold_wall_temperature.assign(-0.01)
        
        
        self.initial_concentration.assign(0.)
        
        self.concentration_rayleigh_number.assign(0.)
        
        self.schmidt_number.assign(1.e32)
        
        self.liquidus_slope.assign(0.)
        
        
        self.timestep_size.assign(10.)
        
        self.temperature_rayleigh_number.assign(3.27e5)
        
        self.prandtl_number.assign(56.2)
        
        self.stefan_number.assign(0.045)
        
        self.regularization_central_temperature_offset.assign(0.01)
        
        self.regularization_smoothing_parameter.assign(0.025)
        
    def initial_values(self):
    
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", 
                 "0.", 
                 "0.", 
                 "(T_h - T_c)*(x[0] < x_m0) + T_c", 
                 "0."),
                T_h = self.hot_wall_temperature, 
                T_c = self.cold_wall_temperature,
                x_m0 = 1./2.**(self.initial_hot_wall_refinement_cycles - 1),
                element = self.element()),
            self.function_space)
            
        return initial_values
        
    def adaptive_goal(self):

        u_t, T_t, C_t, phi_t = self.time_discrete_terms()
        
        dx = self.integration_measure
        
        return -phi_t*dx
        
        
expected_solid_area = 0.552607

def test__cavity_melting_without_concentration__amr__regression__ci__():
    
    sim = CavityMeltingSimulationWithoutConcentration()
    
    sim.assign_initial_values()
    
    for it in range(5):
        
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
    
    p_fine, u_fine, T_fine, C_fine = fenics.split(sim.solution)
    
    phi = sim.semi_phasefield(T = T_fine, C = C_fine)
    
    solid_area = fenics.assemble(phi*fenics.dx)
    
    assert(abs(solid_area - expected_solid_area) < 1.e-6)
    
 
def test__deepcopy__ci__():
    
    tolerance = 1.e-6
    
    sim = CavityMeltingSimulationWithoutConcentration()
    
    sim.assign_initial_values()
    
    for it in range(3):
        
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
        
    sim2 = sim.deepcopy()
    
    assert(all(sim.solution.vector() == sim2.solution.vector()))
    
    for it in range(2):
        
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
    
    p_fine, u_fine, T_fine, C_fine = fenics.split(sim.solution)
    
    phi = sim.semi_phasefield(T = T_fine, C = C_fine)
    
    solid_area = fenics.assemble(phi*fenics.dx)
    
    assert(abs(solid_area - expected_solid_area) < tolerance)
    
    assert(not (sim.solution.vector() == sim2.solution.vector()))
    
    for it in range(2):
        
        sim2.solve(goal_tolerance = 4.e-5)
        
        sim2.advance()
    
    p_fine, u_fine, T_fine, C_fine = fenics.split(sim2.solution)
    
    phi = sim2.semi_phasefield(T = T_fine, C = C_fine)
    
    solid_area = fenics.assemble(phi*fenics.dx)
    
    assert(abs(solid_area - expected_solid_area) < tolerance)
    
    assert(all(sim.solution.vector() == sim2.solution.vector()))
    
    
def test__checkpoint__ci__():
    
    sim = CavityMeltingSimulationWithoutConcentration()
    
    sim.assign_initial_values()
    
    for it in range(2):
        
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
    
    checkpoint_filepath = tempfile.mkdtemp() + "/checkpoint.h5"
    
    sim.write_checkpoint(checkpoint_filepath)
    
    sim2 = CavityMeltingSimulationWithoutConcentration()
    
    sim2.read_checkpoint(checkpoint_filepath)
    
    for it in range(3):
        
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
    
    p_fine, u_fine, T_fine, C_fine = fenics.split(sim.solution)
    
    phi = sim.semi_phasefield(T = T_fine, C = C_fine)
    
    solid_area = fenics.assemble(phi*fenics.dx)
    
    assert(abs(solid_area - expected_solid_area) < 1.e-6)
    
    
def test__coarsen__ci__():
    
    sim = CavityMeltingSimulationWithoutConcentration()
    
    sim.assign_initial_values()
    
    for it in range(3):
        
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
    
    sim.coarsen(absolute_tolerances = (1., 1., 1.e-3, 1., 1.))
    
    for it in range(2):
    
        sim.solve(goal_tolerance = 4.e-5)
        
        sim.advance()
    
    p_fine, u_fine, T_fine, C_fine = fenics.split(sim.solution)
    
    phi = sim.semi_phasefield(T = T_fine, C = C_fine)
    
    solid_area = fenics.assemble(phi*fenics.dx)
    
    tolerance = 1.e-3
    
    assert(abs(solid_area - expected_solid_area) < tolerance)
    