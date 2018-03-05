""" This module runs the benchmark test suite. """
from .context import phaseflow


def test_lid_driven_cavity_benchmark_with_solid_subdomain__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.LDCBenchmarkSimulationWithSolidSubdomain())
    
    
def test_heat_driven_cavity_benchmark__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.HeatDrivenCavityBenchmarkSimulation())
        
    
def test_stefan_problem_benchmark_with_restart__ci__():
    """ This tests that restarting does not affect time accuracy. """
    sim = phaseflow.octadecane_benchmarks.StefanProblemBenchmarkSimulation()
    
    sim.end_time = 0.01
    
    sim.run(verify = False)
    
    sim2 = phaseflow.octadecane_benchmarks.StefanProblemBenchmarkSimulation()
    
    sim2.read_checkpoint(sim.latest_checkpoint_filepath)
    
    assert(sim.state.time == sim2.old_state.time)
    
    assert(all(sim.state.solution.leaf_node().vector() == sim2.old_state.solution.leaf_node().vector()))
    
    sim2.prefix_output_dir_with_tempdir = True
    
    sim2.run(verify = True)
    