from .context import phaseflow

import fenics


def verify_michalek2003_natural_convection_water(w, mesh):
    """Verify directly against steady-state data from michalek2003 published in 

        @article{michalek2003simulations,
          title={Simulations of the water freezing process--numerical benchmarks},
          author={MICHA{\L}EK, TOMASZ and KOWALEWSKI, TOMASZ A},
          journal={Task Quarterly},
          volume={7},
          number={3},
          pages={389--408},
          year={2003}
        }
    """
    verified_solution = {'y': 0.5,
        'x': [0.00, 0.05, 0.12, 0.23, 0.40, 0.59, 0.80, 0.88, 1.00],
        'theta': [1.00, 0.66, 0.56, 0.58, 0.59, 0.62, 0.20, 0.22, 0.00]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_theta in enumerate(verified_solution['theta']):
    
        p = fenics.Point(fenics.Point(verified_solution['x'][i], verified_solution['y']))
        
        if bbt.collides_entity(p):
            
            wval = w(p)
            
            theta = wval[3]
            
            assert(abs(theta - true_theta) < 2.e-2)

            
def test_natural_convection_water__nightly():
    
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
        
        return -q*rho_m*w*abs(T_m - T_ref + theta*(T_c - T_h))**(q - 1.)* \
            fenics.sign(T_m - T_ref + theta*(T_c - T_h))*(T_c - T_h)

            
    beta = 6.91e-5 # [K^-1]
    
    def m_B(T, Ra, Pr, Re):
        
            return Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(rho(theta_f) - rho(T))/rho(theta_f)
            
            
    def ddT_m_B(T, Ra, Pr, Re):
    
        return -Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(ddtheta_rho(T))/rho(theta_f)
        

    w, mesh = phaseflow.run(
        rayleigh_number = Ra,
        prandtl_number = Pr,
        stefan_number = 1.e16,
        temperature_of_fusion = -1.,
        regularization_smoothing_factor = 0.1,
        nlp_relative_tolerance = 1.e-8,
        adaptive = True,
        m_B = m_B,
        ddT_m_B = ddT_m_B,
        mesh = fenics.UnitSquareMesh(m, m, 'crossed'),
        stop_when_steady = True,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            str(theta_hot)+"*near(x[0],  0.) + "+str(theta_cold)+"*near(x[0],  1.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression':str(theta_hot), 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression':str(theta_cold), 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        time_step_size = 0.01,
        start_time = 0.,
        end_time = 2.,
        output_dir = 'output/test_natural_convection_water')
    
    verify_michalek2003_natural_convection_water(w, mesh)

    
if __name__=='__main__':

    test_natural_convection_water__nightly()
    