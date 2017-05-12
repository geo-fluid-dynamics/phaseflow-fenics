import phaseflow

from fenics import UnitSquareMesh

    
def run(linearize = True, adaptive_time = True, m=20, time_step_size = 1.e-3):

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
    def rho(T):
    
        T_maxrho = 4.0293 # [deg C]
        
        rho_m = 999.972 # [kg/m^3]
        
        w = 9.2793e-6 # [(deg C)^(-q)]
        
        q = 1.894816
        
        return rho_m*(1. - w*abs(T - T_maxrho)**q)
        

    def bouyancy_force(theta):
    
        return Ra/(Pr*Re*Re)/(beta*(T_hot - T_cold))*(rho(theta_f) - rho(theta))/rho(theta_f)
    
    
    if linearize:
    
        bc_T_hot = bc_T_cold = 0.    
        
    
        
    phaseflow.run( \
        output_dir = "output/natural_convection_water_linearize"+str(linearize)+"_adaptivetime"+str(adaptive_time)+"_m"+str(m), \
        f_B=bouyancy_force, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = time_step_size, \
        final_time = 10., \
        stop_when_steady = True, \
        linearize = linearize,
        adaptive_time = adaptive_time, \
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