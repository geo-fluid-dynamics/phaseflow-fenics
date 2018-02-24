""" This module runs the benchmark test suite. """
from .context import phaseflow


class BenchmarkSimulationTest:

    def __init__(self, benchmark_simulation):
    
        self.benchmark_simulation = benchmark_simulation
        
        self.benchmark_simulation.prefix_output_dir_with_tempdir = True
        
        
    def run(self):
    
        self.benchmark_simulation.run()
    
        

def test_lid_driven_cavity_benchmark__ci__():
    
    BenchmarkSimulationTestt(phaseflow.octadecane_benchmarks.LidDrivenCavityBenchmarkSimulation()).run()
    

def test_lid_driven_cavity_with_solid_subdomain_benchmark__ci__():
    
    BenchmarkSimulationTest(
        phaseflow.octadecane_benchmarks.LDCBenchmarkSimulationWithSolidSubdomain()).run()
    
    
def test_heat_driven_cavity_benchmark__ci__():
    
    BenchmarkSimulationTest(phaseflow.octadecane_benchmarks.HeatDrivenCavityBenchmarkSimulation()).run()

    
def test_stefan_problem_benchmark__ci__():

    BenchmarkSimulationTest(phaseflow.octadecane_benchmarks.StefanProblemBenchmarkSimulation()).run()
