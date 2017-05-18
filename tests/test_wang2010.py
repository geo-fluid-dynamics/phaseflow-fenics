import phaseflow

from fenics import UnitSquareMesh, Point

    
def test_wang2010_natural_convection_air():

    wang2010_data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    m = 20
    
    w = phaseflow.run( \
        output_dir = "output/test_wang2010_natural_convection", \
        theta_s = -1.,
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = phaseflow.BoundedValue(1.e-3, 1.e-3, 10.), \
        final_time = 10., \
        stop_when_steady = True, \
        steady_relative_tolerance = 1.e-4, \
        linearize = True,
        initial_values_expression = ( \
            "0.", \
            "0.", \
            "0.", \
            "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"), \
        bc_expressions = [ \
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"], \
        [2, "0.", 2, "near(x[0],  0.)", "topological"], \
        [2, "0.", 2, "near(x[0],  1.)", "topological"]])
        
    for i, true_ux in enumerate(wang2010_data['ux']):
        
        wval = w(Point(wang2010_data['x'], wang2010_data['y'][i]))
        
        ux = wval[0]*wang2010_data['Pr']/wang2010_data['Ra']**0.5
        
        assert(abs(ux - true_ux) < 1.e-2)
        
        
if __name__=='__main__':

    test_wang2010_natural_convection_air()