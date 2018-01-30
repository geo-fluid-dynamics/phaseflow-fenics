""" This module runs the benchmark test suite. """
from .context import phaseflow


class BenchmarkTest():

    def __init__(self, benchmark):
    
        self.benchmark = benchmark
        
        self.benchmark.prefix_output_dir_with_tempdir = True
        
        
    def run(self):
    
        self.benchmark.run()    
    
        

def test_adaptive_lid_driven_cavity_benchmark__ci__():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveLidDrivenCavity()).run()
    

def test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark__ci__():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveLidDrivenCavityWithSolidSubdomain()).run()
    
    
def test_adaptive_heat_driven_cavity_benchmark__ci__():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveHeatDrivenCavity()).run()

    
def test_adaptive_heat_driven_cavity_with_water_benchmark():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveHeatDrivenCavityWithWater()).run()
    
    
def test_adaptive_stefan_problem_benchmark__ci__():

    BenchmarkTest(phaseflow.benchmarks.AdaptiveStefanProblem()).run()


def test_adaptive_lid_driven_cavity_benchmark_autodiff():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveLidDrivenCavity(automatic_jacobian = True)).run()
    

def test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark_autodiff():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveLidDrivenCavityWithSolidSubdomain(
        automatic_jacobian = True)).run()
    
    
def test_adaptive_heat_driven_cavity_benchmark_autodiff():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveHeatDrivenCavity(automatic_jacobian = True)).run()

    
def test_adaptive_heat_driven_cavity_with_water_benchmark_autodiff():
    
    BenchmarkTest(phaseflow.benchmarks.AdaptiveHeatDrivenCavityWithWater(automatic_jacobian = True)).run()
    
    
def test_adaptive_stefan_problem_benchmark_autodiff():

    BenchmarkTest(phaseflow.benchmarks.AdaptiveStefanProblem(automatic_jacobian = True)).run()
