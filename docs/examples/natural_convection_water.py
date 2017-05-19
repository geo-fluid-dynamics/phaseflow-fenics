import phaseflow

from fenics import UnitSquareMesh

from ufl import sign

    
def run(m=20):
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
    
    ''' @todo Verify usage of equation (24) and (25) from danaila2014newton
    
    The way that Danaila wrote down the density and bouyancy force equations (24) and (25) confuses me. (24) writes density as a function of temperature T, while (25) uses the density as a function of the normalized temperature, i.e. rho(theta). Furthermore, (25) has the bouyancy force as a function of the normalized temperature, i.e. f_b(theta), and it is expressed both with the temperatures T_h and T_c as well as the normalized temperatures theta and theta_f.

    ''' 
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

    phaseflow.run( \
        output_dir = "output/natural_convection_water_linearize"+str(linearize)+"_m"+str(m), \
        Ra = Ra, \
        Pr = Pr, \
        m_B = lambda theta : Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(rho(theta_f) - rho(theta))/rho(theta_f), \
        dm_B_dtheta = lambda theta : -Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(ddtheta_rho(theta))/rho(theta_f), \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = phaseflow.BoundedValue(0.001, 0.004, 10.), \
        final_time = 10., \
        linearize = linearize, \
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

    run()