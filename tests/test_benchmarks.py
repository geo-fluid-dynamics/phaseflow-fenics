""" This module runs the test suite.

@todo 

    Presently this runs all of the benchmarks as tests.
    Some of the benchmarks take a longer-than-desireable amount of time to run for routine testing.
    Perhaps each benchmark should have arguments which allow us to run quick versions in the test suite. 
"""
from .context import phaseflow


def test_lid_driven_cavity_benchmark():
    
    phaseflow.benchmarks.LidDrivenCavity().run()


def test_adaptive_lid_driven_cavity_benchmark():
    
    phaseflow.benchmarks.AdaptiveLidDrivenCavity().run()


def test_lid_driven_cavity_with_solid_subdomain_benchmark():
    
    phaseflow.benchmarks.LidDrivenCavityWithSolidSubdomain().run()
    

def test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark():
    
    phaseflow.benchmarks.AdaptiveLidDrivenCavityWithSolidSubdomain().run()
    
    
def test_heat_driven_cavity_benchmark():
    
    phaseflow.benchmarks.HeatDrivenCavity().run()    
    
    
def test_adaptive_heat_driven_cavity_benchmark():
    
    phaseflow.benchmarks.AdaptiveHeatDrivenCavity().run()    
    
    
if __name__=='__main__':

    test_lid_driven_cavity_benchmark()
    
    test_adaptive_lid_driven_cavity_benchmark()
    
    test_adaptive_lid_driven_cavity_with_solid_subdomain_benchmark()
    
    test_heat_driven_cavity_benchmark()
    
    test_adaptive_heat_driven_cavity_benchmark()
