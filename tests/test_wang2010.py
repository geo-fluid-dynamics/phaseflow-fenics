from .context import phaseflow

import fenics


def verify_against_wang2010(w):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    for i, true_ux in enumerate(data['ux']):
        
        wval = w(fenics.Point(data['x'], data['y'][i]))
        
        ux = wval[0]*data['Pr']/data['Ra']**0.5
        
        assert(abs(ux - true_ux) < 2.e-2)
        

def test_wang2010_natural_convection_air():

    wang2010_data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    m = 20
    
    w = phaseflow.run(
        mesh = fenics.UnitSquareMesh(m, m, 'crossed'),
        time_step_bounds = (1.e-3, 1.e-3, 10.),
        final_time = 10.,
        stop_when_steady = True,
        steady_relative_tolerance = 1.e-4,
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
        output_dir = "output/test_wang2010_natural_convection")
        
    verify_against_wang2010(w)


if __name__=='__main__':
    
    test_wang2010_natural_convection_air()