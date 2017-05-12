import phaseflow

from fenics import UnitSquareMesh

from ufl import *

    
def run(adaptive_space = False, m=20, adaptive_time = True, time_step_size = 1.e-3):

    linearize = True
    
    Ra = 2.518084e6
    
    Pr = 6.99
    
    T_hot = 10. # [deg C]
    
    T_cold = 0. # [deg C]
    
    bc_T_hot = T_hot
    
    bc_T_cold = T_cold
    
    T_f = T_fusion = 0. # [deg C]
    
    T_ref = T_fusion
    
    theta_f = theta_fusion = T_fusion - T_ref
    
    ''' @todo Verify usage of equation (24) and (25) from danaila2014newton
    
    The way that Danaila wrote down the density and bouyancy force equations (24) and (25) confuses me. (24) writes density as a function of temperature T, while (25) uses the density as a function of the normalized temperature, i.e. rho(theta). Furthermore, (25) has the bouyancy force as a function of the normalized temperature, i.e. f_b(theta), and it is expressed both with the temperatures T_h and T_c as well as the normalized temperatures theta and theta_f.
    
    def theta(T):
    
        T_ref = T_fusion
        
        return (T - T_ref)/(T_hot - T_cold)
    ''' 
    T_maxrho = 4.0293 # [deg C]
        
    rho_m = 999.972 # [kg/m^3]
    
    w = 9.2793e-6 # [(deg C)^(-q)]
    
    q = 1.894816
    
    def rho(T):            
        
        return rho_m*(1. - w*abs(T - T_maxrho)**q)
    
    # @todo define ddT_rho(T) here instead of providing the long form below. I can do this with sympy.

    Re = phaseflow.Re
    
    beta = 6.91e-5 # [K^-1]
    
    if linearize:
    
        bc_T_hot = bc_T_cold = 0.  
            
    g = (0., -1.)
            
    phaseflow.run( \
        output_dir = "output/natural_convection_water_linearize"+str(linearize)+"_adaptivetime"+str(adaptive_time)+"_m"+str(m), \
        Ra = Ra, \
        Pr = Pr, \
        g = g, \
        m_B = lambda theta : Ra/(Pr*Re*Re)/(beta*(T_hot - T_cold))*(rho(theta_f) - rho(theta))/rho(theta_f), \
        dm_B_dtheta = lambda theta : -(Ra*q*w*abs(T_maxrho - theta)**(q - 1.)*sign(T_maxrho - theta))/(Pr*Re**2*beta*(T_cold - T_hot)*(w*abs(T_maxrho - theta_f)**q - 1.)), \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = time_step_size, \
        final_time = 10., \
        stop_when_steady = True, \
        linearize = linearize, \
        adaptive_time = adaptive_time, \
        adaptive_space = adaptive_space, \
        initial_values_expression = ( \
            "0.", \
            "0.", \
            "0.", \
            str(T_hot)+"*near(x[0],  0.) + "+str(T_cold)+"*near(x[0],  1.)"), \
        bc_expressions = [ \
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"], \
        [2, str(bc_T_hot), 2, "near(x[0],  0.)", "topological"], \
        [2, str(bc_T_cold), 2, "near(x[0],  1.)", "topological"]])

        
if __name__=='__main__':

    run()