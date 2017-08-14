from .context import phaseflow

import fenics


def test_only_write_initial_values():
    
    w, mesh = phaseflow.run(output_dir='output/test_only_write_initial_values', final_time=0.)
    
    
if __name__=='__main__':

    test_only_write_initial_values()