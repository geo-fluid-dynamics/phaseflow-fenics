""" This module runs the benchmark test suite. """
from .context import phaseflow


def test_adaptive_lid_driven_cavity_benchmark():
    
    phaseflow.benchmarks.AdaptiveLidDrivenCavity().run()


def test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark():
    
    phaseflow.benchmarks.AdaptiveLidDrivenCavityWithSolidSubdomain().run()

    
def test_adaptive_heat_driven_cavity_benchmark():
    
    phaseflow.benchmarks.AdaptiveHeatDrivenCavity().run()
    

def test_adaptive_heat_driven_cavity_with_water_benchmark():
    
    phaseflow.benchmarks.AdaptiveHeatDrivenCavityWithWater().run()    
    
    
def test_adaptive_stefan_problem_benchmark():

    phaseflow.benchmarks.AdaptiveStefanProblem().run()
