from .context import phaseflow

import fenics
import pytest


def verify_against_wang2010(w, mesh):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
        
            ux = wval[0]*data['Pr']/data['Ra']**0.5
        
            assert(abs(ux - true_ux) < 2.e-2)
        
        
def wang2010_natural_convection_air(output_dir='output/test_wang2010_natural_convection_air', final_time=10., restart=False,
        automatic_jacobian=True):

    m = 20
    
    theta_hot = 0.5
    
    theta_cold = -0.5
    
    w, mesh = phaseflow.run(
        Ste = 1.e16,
        mesh = fenics.UnitSquareMesh(m, m, 'crossed'),
        time_step_bounds = (1.e-3, 1.e-3, 0.01),
        final_time = final_time,
        output_times = ('initial', 1.e-3, 0.01, 0.1, 1., 'final',),
        stop_when_steady = True,
        automatic_jacobian = automatic_jacobian,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            str(theta_hot)+"*near(x[0],  0.) + "+str(theta_cold)+"*near(x[0],  1.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_hot), 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_cold), 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        output_dir = output_dir,
        restart = restart)
        
    return w, mesh
    
    
def test_debug_wang2010_natural_convection_air_autoJ():
    
    w, mesh = wang2010_natural_convection_air(automatic_jacobian=True)
        
    verify_against_wang2010(w, mesh)
    
    
@pytest.mark.dependency()
def test_wang2010_natural_convection_air_manualJ():
    
    w, mesh = wang2010_natural_convection_air(automatic_jacobian=False)
        
    verify_against_wang2010(w, mesh)
    

@pytest.mark.dependency(depends=["test_wang2010_natural_convection_air_manualJ"])
def test_wang2010_restart():

    w, mesh = wang2010_natural_convection_air(final_time=10.5, restart=True, automatic_jacobian=False)
        
    verify_against_wang2010(w, mesh)


def verify_regression_water(w, mesh):
    ''' Comparing directly to the steady state benchmark data 
    from Michalek2003 in this case would take longer 
    than we should spend on a this part of the test suite. 
    So instead we subsitute this cheaper regression test.'''
    verified_solution = {'y': 0.5, 'x': [0.00, 0.10, 0.20, 0.30, 0.40, 0.70, 0.90, 1.00], 'theta': [1.000, 0.551, 0.202, 0.166, 0.211, 0.209, 0.238, 0.000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_theta in enumerate(verified_solution['theta']):
    
        p = fenics.Point(fenics.Point(verified_solution['x'][i], verified_solution['y']))
        
        if bbt.collides_entity(p):
            
            wval = w(p)
            
            theta = wval[3]
            
            assert(abs(theta - true_theta) < 1.e-2)

            
def regression_natural_convection_water(automatic_jacobian=False):
    m = 10

    Ra = 2.518084e6
    
    Pr = 6.99
    
    T_h = 10. # [deg C]
    
    T_c = 0. # [deg C]
    
    theta_hot = 1.
    
    theta_cold = 0.
    
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
        
        return -q*rho_m*w*abs(T_m - T_ref + theta*(T_c - T_h))**(q - 1.)*fenics.sign(T_m - T_ref + theta*(T_c - T_h))*(T_c - T_h)

    Re = phaseflow.globals.Re
    
    beta = 6.91e-5 # [K^-1]

    w, mesh = phaseflow.run(
        Ra = Ra,
        Pr = Pr,
        Ste = 1.e16,
        m_B = lambda theta : Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(rho(theta_f) - rho(theta))/rho(theta_f),
        ddtheta_m_B = lambda theta : -Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(ddtheta_rho(theta))/rho(theta_f),
        mesh = fenics.UnitSquareMesh(m, m, 'crossed'),
        time_step_bounds = (0.001, 0.005, 0.005),
        final_time = 0.18,
        output_times = (),
        automatic_jacobian = False,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            str(theta_hot)+"*near(x[0],  0.) + "+str(theta_cold)+"*near(x[0],  1.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression':str(theta_hot), 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression':str(theta_cold), 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        output_dir = 'output/test_regression_natural_convection_water')
        
    verify_regression_water(w, mesh)


def test_debug_regression_natural_convection_water_autoJ():

    regression_natural_convection_water(automatic_jacobian=True)
    

def test_debug_regression_natural_convection_water_manualJ():
    
    regression_natural_convection_water(automatic_jacobian=False)
    
    
if __name__=='__main__':

    test_debug_wang2010_natural_convection_air_autoJ()
    
    test_wang2010_natural_convection_air_manualJ()
    
    test_wang2010_restart()
    
    test_debug_regression_natural_convection_water_autoJ()
    
    test_debug_regression_natural_convection_water_manualJ()
