from .context import phaseflow


def test_lid_driven_cavity():
    
    phaseflow.benchmarks.LidDrivenCavity().run()
    
    
if __name__=='__main__':

    test_lid_driven_cavity()

