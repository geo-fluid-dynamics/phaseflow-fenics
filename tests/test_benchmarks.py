""" This module runs the benchmark test suite. """
from .context import phaseflow


def test_adaptive_lid_driven_cavity_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveLidDrivenCavity()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()


def test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveLidDrivenCavityWithSolidSubdomain()

    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    
def test_adaptive_heat_driven_cavity_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveHeatDrivenCavity()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()

    
def test_adaptive_heat_driven_cavity_with_water_benchmark():
    
    benchmark = phaseflow.benchmarks.AdaptiveHeatDrivenCavityWithWater()
    
    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    
    
def test_adaptive_stefan_problem_benchmark():

    benchmark = phaseflow.benchmarks.AdaptiveStefanProblem()

    benchmark.prefix_output_dir_with_tempdir = True
    
    benchmark.run()
    