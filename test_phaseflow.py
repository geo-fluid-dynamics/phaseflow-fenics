import phaseflow

from fenics import UnitSquareMesh, Point

    
def test_ghia1982_steady_lid_driven_cavity():

    ghia1982_data = {'Re': 100, 'x': 0.5, 'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000], 'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
        
    v = 1.
    
    m = 20

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'

    w = phaseflow.run(linearize = False, \
        adaptive_time = False, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1.e12, \
        time_step_size = 1.e12, \
        mu = 0.01, \
        output_dir="output/test_ghia1982_steady_lid_driven_cavity", \
        s_theta ='0.', \
        initial_values_expression = ('0.', '0.', '0.', '0.'), \
        bc_expressions = [[0, (str(v), '0.'), 3, lid, "topological"], [0, ('0.', '0.'), 3, fixed_walls, "topological"], [1, '0.', 2, bottom_left_corner, "pointwise"]])

    for i, true_ux in enumerate(ghia1982_data['ux']):
        
        wval = w(Point(ghia1982_data['x'], ghia1982_data['y'][i]))
        
        ux = wval[0]
        
        assert(abs(ux - true_ux) < 1.e-2)
        
    
if __name__=='__main__':

    test_ghia1982_steady_lid_driven_cavity()