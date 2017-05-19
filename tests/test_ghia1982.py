import phaseflow

from fenics import UnitSquareMesh, Point

lid = 'near(x[1],  1.)'

fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'


def verify_against_ghia1982(w):

    data = {'Re': 100, 'x': 0.5, 'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000], 'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
    
    for i, true_ux in enumerate(data['ux']):
        
        wval = w(Point(data['x'], data['y'][i]))
        
        ux = wval[0]
        
        assert(abs(ux - true_ux) < 2.e-2)
        

def test_ghia1982_steady_lid_driven_cavity():
   
    m = 20

    w = phaseflow.run(linearize = False,
        mesh = UnitSquareMesh(m, m, "crossed"),
        final_time = 1.e12,
        time_step_size = phaseflow.BoundedValue(1.e12, 1.e12, 1.e12),
        mu_l = 0.01,
        theta_s = -1.,
        output_dir="output/test_ghia1982_steady_lid_driven_cavity",
        initial_values_expression = (lid, '0.', '0.', '0.'),
        bc_expressions = [[0, ('1.', '0.'), 3, lid, "topological"], [0, ('0.', '0.'), 3, fixed_walls, "topological"], [1, '0.', 2, bottom_left_corner, "pointwise"]])

    verify_against_ghia1982(w)
        
        
def test_ghia1982_steady_lid_driven_cavity_linearized():

    m = 20

    w = phaseflow.run(linearize = True,
        mesh = UnitSquareMesh(m, m, "crossed"),
        final_time = 1.e12,
        time_step_size = phaseflow.BoundedValue(1.e12, 1.e12, 1.e12),
        mu_l = 0.01,
        theta_s = -1.,
        output_dir="output/test_ghia1982_steady_lid_driven_cavity_linearized",
        initial_values_expression = (lid, '0.', '0.', '0.'),
        bc_expressions = [[0, ('0.', '0.'), 3, lid, "topological"], [0, ('0.', '0.'), 3, fixed_walls, "topological"], [1, '0.', 2, bottom_left_corner, "pointwise"]])

    verify_against_ghia1982(w)

def test_ghia1982_steady_lid_driven_cavity_amr():

    coarse_m = 4

    w = phaseflow.run(linearize = False,
        mesh = UnitSquareMesh(coarse_m, coarse_m, "crossed"),
        adaptive_space = True,
        adaptive_space_error_tolerance = 1.e-4,
        final_time = 1.e12,
        time_step_size = phaseflow.BoundedValue(1.e12, 1.e12, 1.e12),
        mu_l = 0.01,
        theta_s = -1.,
        output_dir="output/test_ghia1982_steady_lid_driven_cavity_amr",
        initial_values_expression = (lid, '0.', '0.', '0.'),
        bc_expressions = [[0, ('1.', '0.'), 3, 'near(x[1],  1.)', "topological"], [0, ('0.', '0.'), 3, 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)', "topological"], [1, '0.', 2, 'near(x[0], 0.) && near(x[1], 0.)', "pointwise"]])

    verify_against_ghia1982(w)
    

def test_ghia1982_steady_lid_driven_cavity_linearized_amr():

    coarse_m = 4

    w = phaseflow.run( linearize = True,
        mesh = UnitSquareMesh(coarse_m, coarse_m, "crossed"),
        adaptive_space = True,
        adaptive_space_error_tolerance = 1.e-4,
        final_time = 1.e12,
        time_step_size = phaseflow.BoundedValue(1.e12, 1.e12, 1.e12),
        mu_l = 0.01,
        theta_s = -1.,
        output_dir="output/test_ghia1982_steady_lid_driven_cavity_linearized_amr",
        initial_values_expression = (lid, '0.', '0.', '0.'),
        bc_expressions = [[0, ('0.', '0.'), 3, 'near(x[1],  1.)', "topological"], [0, ('0.', '0.'), 3, 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)', "topological"], [1, '0.', 2, 'near(x[0], 0.) && near(x[1], 0.)', "pointwise"]])

    verify_against_ghia1982(w)
    
        
if __name__=='__main__':

    test_ghia1982_steady_lid_driven_cavity()
    
    test_ghia1982_steady_lid_driven_cavity_linearized()
    
    test_ghia1982_steady_lid_driven_cavity_amr()
    
    #test_ghia1982_steady_lid_driven_cavity_linearized_amr()