import fenics
import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import phaseflow


def wang2010_natural_convection_air(output_dir='output/test_wang2010_natural_convection', final_time=10., restart=False):

    m = 20
    
    w, mesh = phaseflow.run(
        Ste = 1.e16,
        mesh = fenics.UnitSquareMesh(m, m, 'crossed'),
        time_step_bounds = (1.e-3, 1.e-3, 10.),
        final_time = final_time,
        output_times = ('final',),
        stop_when_steady = True,
        linearize = True,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
        boundary_conditions = [
        {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': "topological"},
        {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
        {'subspace': 2, 'value_expression': "0.", 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        output_format = 'xdmf',
        output_dir = output_dir,
        restart = restart)
        
    return w, mesh
        

def wang2010_natural_convection_air_restart():
    
    m = 20
        
    output_dir = 'output/test_wang2010_natural_convection_restart'
    
    w, mesh = wang2010_natural_convection_air(output_dir=output_dir, final_time=5., restart=False)
    
    w, mesh = wang2010_natural_convection_air(output_dir=output_dir, final_time=10., restart=True)
        
    
if __name__=='__main__':
    
    wang2010_natural_convection_air_restart()
