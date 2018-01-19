from .context import phaseflow


def test_lid_driven_cavity_benchmark():
    
    phaseflow.benchmarks.LidDrivenCavity().run()


def test_heat_driven_cavity_benchmark():
    
    phaseflow.benchmarks.HeatDrivenCavity().run()    
    
    
def test_adaptive_heat_driven_cavity_benchmark():
    
    phaseflow.benchmarks.AdaptiveHeatDrivenCavity().run()    
    
    
if __name__=='__main__':

    test_lid_driven_cavity_benchmark()
    
    test_heat_driven_cavity_benchmark()
    
    test_adaptive_heat_driven_cavity_benchmark()
