""" This module runs the benchmark test suite. """
from .context import phaseflow


def test_lid_driven_cavity_benchmark__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.LidDrivenCavityBenchmarkSimulation())
    

def test_lid_driven_cavity_benchmark_with_solid_subdomain__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.LDCBenchmarkSimulationWithSolidSubdomain())
    
    
def test_heat_driven_cavity_benchmark__ci__():
    
    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.HeatDrivenCavityBenchmarkSimulation())

    
def test_stefan_problem_benchmark__ci__():

    phaseflow.helpers.run_simulation_with_temporary_output(
        phaseflow.octadecane_benchmarks.StefanProblemBenchmarkSimulation())
