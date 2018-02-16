"""**pure.py** contains code specific to pure materials.

For example. this is useful for pure water, but not for salt water.
"""
import fenics
import phaseflow


def make_mixed_element(mesh_ufl_cell, pressure_degree = 1, temperature_degree = 1):
    """ 
    
    An attempt was made to define this as a sub-class of fenics.MixedElement,
    instead of as a function;
    but DOLFIN's JIT compiler was throwing an error related to its __repr attribute.
    """
    pressure_element = fenics.FiniteElement("P", mesh_ufl_cell, pressure_degree)
    
    velocity_degree = pressure_degree + 1
    
    velocity_element = fenics.VectorElement("P", mesh_ufl_cell, velocity_degree)

    temperature_element = fenics.FiniteElement("P", mesh_ufl_cell, temperature_degree)
    
    return fenics.MixedElement([pressure_element, velocity_element, temperature_element])


class PhaseDependentMaterialProperty(phaseflow.core.ContinuousFunction):

    def __init__(self, liquid_value, solid_value):
    
        P_L = fenics.Constant(liquid_value)
        
        P_S = fenics.Constant(solid_value)
        
        def P(semi_phasefield_mapping, T):
        
            phi = semi_phasefield_mapping.function
            
            return P_L + (P_S - P_L)*phi(T)
        
        
        def dP(semi_phasefield_mapping, T):
        
            dphi = semi_phasefield_mapping.derivative_function
            
            return (P_S - P_L)*dphi(T)
            
        phaseflow.core.ContinuousFunction.__init__(self, function = P, derivative_function = dP)
            
            
class IdealizedLinearBoussinesqBuoyancy(phaseflow.core.ContinuousFunction):
    
    def __init__(self, rayleigh_numer, prandtl_number, gravity = [0., -1.]):
    
        g = fenics.Constant(gravity)
        
        Ra = fenics.Constant(rayleigh_numer)
        
        Pr = fenics.Constant(prandtl_number)
        
        Re = fenics.Constant(1.)
        
        def f_B(T):
        
            return T*Ra/(Pr*Re**2)*g
    
    
        def df_B(T):
            """In this case the derivative df_B is no longer temperature-dependent,
            but we still define this as a function of temperature so that the 
            interface is consistent.
            """

            return Ra/(Pr*Re**2)*g
    
        phaseflow.core.ContinuousFunction.__init__(self, function=f_B, derivative_function=df_B)
        
        
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
        
        
class TanhSemiPhasefieldMapping(phaseflow.core.ContinuousFunction):

    def __init__(self, regularization_smoothing_parameter, regularization_central_temperature = 0.):
    
        T_r = fenics.Constant(regularization_central_temperature)
        
        r = fenics.Constant(regularization_smoothing_parameter)
    
        def phi(T):

            return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
        def sech(theta):
        
            return 1./fenics.cosh(theta)
    
    
        def dphi(T):
        
            return -sech((T_r - T)/r)**2/(2.*r)
            
        phaseflow.core.ContinuousFunction.__init__(self, function=phi, derivative_function=dphi)
    
    
class ConstantFunction(phaseflow.core.ContinuousFunction):

    def __init__(self, constant_value):
    
        constant = fenics.Constant(constant_value)
        
        def f(T):
    
            return constant
        
        def df(T):
        
            return 0.*constant
            
        phaseflow.core.ContinuousFunction.__init__(self, function=f, derivative_function=df)
 