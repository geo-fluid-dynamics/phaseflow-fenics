""" This module runs the benchmark test suite. """
from .context import phaseflow


def test_adaptive_lid_driven_cavity_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveLidDrivenCavity()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()


def test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveLidDrivenCavityWithSolidSubdomain().run()

    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    
def test_adaptive_heat_driven_cavity_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveHeatDrivenCavity().run()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()

    
def test_adaptive_heat_driven_cavity_with_water_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveHeatDrivenCavityWithWater().run()    
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    
def test_adaptive_stefan_problem_benchmark():

    benchmark = phaseflow.benchmarks.AdaptiveStefanProblem().run()

    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    