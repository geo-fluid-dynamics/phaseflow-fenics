import phaseflow

from fenics import UnitSquareMesh, Point

from ufl import sign

''' @todo For now these are being used as a very quick regression test,
making sure that these simply run and converge.'''

def test_quick_regression_natural_convection_air():

    m = 10
    
    w = phaseflow.run( \
        output_dir = "output/test_wang2010_natural_convection", \
        theta_s = -1.,
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = phaseflow.BoundedValue(1.e-3, 1.e-3, 10.), \
        final_time = 3.e-3, \
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
        
        
def test_quick_regression_natural_convection_water():

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
        output_dir = "output/test_quick_regression_natural_convection_water", \
        Ra = Ra, \
        Pr = Pr, \
        theta_s = -1.,
        m_B = lambda theta : Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(rho(theta_f) - rho(theta))/rho(theta_f), \
        dm_B_dtheta = lambda theta : -Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(ddtheta_rho(theta))/rho(theta_f), \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = phaseflow.BoundedValue(0.005, 0.005, 0.01), \
        final_time = 0.015, \
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
        
        
if __name__=='__main__':

    test_quick_regression_natural_convection_air()
    
    test_quick_regression_natural_convection_water()