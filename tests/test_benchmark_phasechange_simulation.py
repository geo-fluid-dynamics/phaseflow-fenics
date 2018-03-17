""" This module runs the benchmark test suite. """
from .context import phaseflow
import fenics


def test_lid_driven_cavity_benchmark_with_solid_subdomain__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.benchmark_phasechange_simulation.LDCBenchmarkPhaseChangeSimulationWithSolidSubdomain())
    
    
def test_heat_driven_cavity_benchmark__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.benchmark_phasechange_simulation.HeatDrivenCavityBenchmarkPhaseChangeSimulation())
        
    
def test_stefan_problem_benchmark_with_restart__ci__():
    """ This tests that restarting does not affect time accuracy. """
    sim = phaseflow.benchmark_phasechange_simulation.StefanProblemBenchmarkPhaseChangeSimulation()
    
    sim.end_time = 0.01
    
    sim.prefix_output_dir_with_tempdir = True
    
    sim.run(verify = False)
    
    sim2 = phaseflow.benchmark_phasechange_simulation.StefanProblemBenchmarkPhaseChangeSimulation()
    
    sim2.read_checkpoint(sim.latest_checkpoint_filepath)
    
    assert(sim.state.time == sim2.old_state.time)
    
    assert(all(sim.state.solution.leaf_node().vector() == sim2.old_state.solution.leaf_node().vector()))
    
    sim2.prefix_output_dir_with_tempdir = True
    
    sim2.run(verify = True)
    
    
class RegressionTestSimulation( \
        phaseflow.benchmark_phasechange_simulation.ConvectionCoupledMeltingOctadecanePCMBenchmarkPCSimulation):
    """ This modifies the phasechange_simulation melting benchmark for quick regression testing. """
    def __init__(self):
        
        phaseflow.benchmark_phasechange_simulation.ConvectionCoupledMeltingOctadecanePCMBenchmarkPCSimulation.__init__(self)
        
        self.timestep_size = 10.
        
        self.end_time = 30.
        
        self.quadrature_degree = 8
        
        self.mesh_size = (1, 1)
        
        self.initial_hot_wall_refinement_cycles = 6
        
        self.adaptive_goal_tolerance = 1.e-5
    
        self.output_dir += "regression/"
        
        
    def verify(self):
        """ Test regression based on a previous solution from Phaseflow.
        
        In Paraview, the $T = 0.01$ (i.e. the regularization_central_temperature) contour was drawn
        at time $t = 30.$ (i.e. the end_time).
        
        A point from this contour in the upper portion of the domain, 
        where the PCI has advanced more quickly, was recorded to be (0.278, 0.875).
        This was checked for commit a8a8a039e5b89d71b6cceaef519dfbf136322639.
        
        Here we verify that the temperature is above the melting temperature left of the expected PCI,
        which is advancing to the right, and is below the melting temperature right of the expected PCI.
        """
        pci_y_position_to_check =  0.88
        
        reference_pci_x_position = 0.28
        
        position_offset = 0.01
        
        left_temperature = self.state.solution.leaf_node()(
            fenics.Point(reference_pci_x_position - position_offset, pci_y_position_to_check))[3]
        
        right_temperature = self.state.solution.leaf_node()(
            fenics.Point(reference_pci_x_position + position_offset, pci_y_position_to_check))[3]
        
        assert((left_temperature > self.regularization_central_temperature) 
            and (self.regularization_central_temperature > right_temperature))
            

def test_convection_coupled_melting_phasechange_simulation_pcm_regression__ci__():

    phaseflow.helpers.run_simulation_with_temporary_output(RegressionTestSimulation())
    