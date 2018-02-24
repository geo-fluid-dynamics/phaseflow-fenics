class GebhartWaterBuoyancy(phaseflow.core.ContinuousFunction):
    """ Water buoyancy model centered around the density anomaly, published in \cite{gebhart1977}. """
    def __init__(self, hot_temperature, cold_temperature, rayleigh_numer, prandtl_number, 
            gravity = [0., -1.]):
    
        T_hot = fenics.Constant(hot_temperature)
        
        T_cold = fenics.Constant(cold_temperature)
        
        Ra = fenics.Constant(rayleigh_numer)
        
        Pr = fenics.Constant(prandtl_number)
        
        Re = fenics.Constant(1.)
        
        g = fenics.Constant(gravity)
        
        T_m = fenics.Constant(4.0293)  # [deg C]
        
        T_fusion = fenics.Constant(0.)  # [deg C]
        
        T_ref = T_fusion
        
        scaled_T_fusion = fenics.Constant(0.)
        
        rho_m = fenics.Constant(999.972)  # [kg/m^3]
        
        w = fenics.Constant(9.2793e-6)  # [(deg C)^(-q)]
        
        q = fenics.Constant(1.894816)
        
        def rho(scaled_T):            
        
            return rho_m*(1. - w*abs((T_hot - T_cold)*scaled_T + T_ref - T_m)**q)
        
        
        def drho(scaled_T):
            
            return -q*rho_m*w*abs(T_m - T_ref + scaled_T*(T_cold - T_hot))**(q - 1.)* \
                fenics.sign(T_m - T_ref + scaled_T*(T_cold - T_hot))*(T_cold - T_hot)
        
        
        beta = fenics.Constant(6.91e-5)  # [K^-1]
        
        def f_B(T):
        
            return Ra/(Pr*Re*Re)/(beta*(T_hot - T_cold))* \
                (rho(scaled_T_fusion) - rho(T))/rho(scaled_T_fusion)*g
            
            
        def df_B(T):
        
            return -Ra/(Pr*Re*Re)/(beta*(T_hot - T_cold))*(drho(T))/rho(scaled_T_fusion)*g
        
        
        phaseflow.core.ContinuousFunction.__init__(self, function=f_B, derivative_function=df_B)