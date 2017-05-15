import phaseflow

from fenics import UnitSquareMesh, Point

from ufl import sign

    
def test_ghia1982_steady_lid_driven_cavity():

    ghia1982_data = {'Re': 100, 'x': 0.5, 'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000], 'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
        
    v = 1.
    
    m = 20

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'

    w = phaseflow.run(linearize = False, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1.e12, \
        time_step_size = phaseflow.BoundedValue(1.e12, 1.e12, 1.e12), \
        mu = 0.01, \
        output_dir="output/test_ghia1982_steady_lid_driven_cavity", \
        s_theta ='0.', \
        initial_values_expression = ('0.', '0.', '0.', '0.'), \
        bc_expressions = [[0, (str(v), '0.'), 3, lid, "topological"], [0, ('0.', '0.'), 3, fixed_walls, "topological"], [1, '0.', 2, bottom_left_corner, "pointwise"]])

    for i, true_ux in enumerate(ghia1982_data['ux']):
        
        wval = w(Point(ghia1982_data['x'], ghia1982_data['y'][i]))
        
        ux = wval[0]
        
        assert(abs(ux - true_ux) < 1.e-2)
        

def test_wang2010_natural_convection_air():

    wang2010_data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    m = 40
    
    w = phaseflow.run( \
        output_dir = "output/test_wang2010_natural_convection", \
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
        
        
def test_regression_natural_convection_water():

    ''' Comparing directly to the steady state benchmark data 
    from Michalek2003 in this case would take longer 
    than we should spend on a this part of the test suite. 
    So instead we subsitute this cheaper regression test.'''
    verified_solution = {'y': 0.5, 'x': [0.00, 0.10, 0.20, 0.30, 0.40, 0.70, 0.90, 1.00], 'theta': [1.000, 0.551, 0.202, 0.166, 0.211, 0.209, 0.238, 0.000]}
    
    m = 10

    linearize = True
    
    Ra = 2.518084e6
    
    Pr = 6.99
    
    T_h = 10. # [deg C]
    
    T_c = 0. # [deg C]
    
    theta_hot = 1.
    
    theta_cold = 0.
    
    bc_theta_hot = theta_hot
    
    bc_theta_cold = theta_cold
    
    T_f = T_fusion = 0. # [deg C]
    
    T_ref = T_fusion
    
    theta_f = theta_fusion = T_fusion - T_ref
    
    T_m = 4.0293 # [deg C]
        
    rho_m = 999.972 # [kg/m^3]
    
    w = 9.2793e-6 # [(deg C)^(-q)]
    
    q = 1.894816
    
    def rho(theta):            
        
        return rho_m*(1. - w*abs((T_h - T_c)*theta + T_ref - T_m)**q)
        
    def ddtheta_rho(theta):
        
        return -q*rho_m*w*abs(T_m - T_ref + theta*(T_c - T_h))**(q - 1.)*sign(T_m - T_ref + theta*(T_c - T_h))*(T_c - T_h)

    Re = phaseflow.Re
    
    beta = 6.91e-5 # [K^-1]
    
    if linearize:
    
        bc_theta_hot = bc_theta_cold = 0.  

    w = phaseflow.run( \
        output_dir = "output/test_regression_natural_convection_water", \
        Ra = Ra, \
        Pr = Pr, \
        m_B = lambda theta : Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(rho(theta_f) - rho(theta))/rho(theta_f), \
        dm_B_dtheta = lambda theta : -Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(ddtheta_rho(theta))/rho(theta_f), \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = phaseflow.BoundedValue(0.005, 0.005, 0.01), \
        final_time = 0.18, \
        linearize = linearize, \
        newton_relative_tolerance = 1.e-4, \
        max_newton_iterations = 10, \
        initial_values_expression = ( \
            "0.", \
            "0.", \
            "0.", \
            str(theta_hot)+"*near(x[0],  0.) + "+str(theta_cold)+"*near(x[0],  1.)"), \
        bc_expressions = [ \
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"], \
        [2, str(bc_theta_hot), 2, "near(x[0],  0.)", "topological"], \
        [2, str(bc_theta_cold), 2, "near(x[0],  1.)", "topological"]])
        
    for i, true_theta in enumerate(verified_solution['theta']):
        
        wval = w(Point(verified_solution['x'][i], verified_solution['y']))
        
        theta = wval[3]
        
        assert(abs(theta - true_theta) < 1.e-2)
        
        
if __name__=='__main__':

    test_ghia1982_steady_lid_driven_cavity()
    
    test_wang2010_natural_convection()