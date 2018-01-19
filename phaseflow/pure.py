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


class PhaseDependentMaterialProperty(phaseflow.ContinuousFunction):

    def __init__(self, semi_phasefield_mapping, liquid_value, solid_value):
    
        P_L = fenics.Constant(liquid_value)
        
        P_S = fenics.Constant(solid_value)
        
        phi = semi_phasefield_mapping.function
        
        def P(T):
        
            return P_L + (P_S - P_L)*phi(T)
        

        dphi = semi_phasefield_mapping.derivative_function
        
        def dP(T):
        
            return (P_S - P_L)*dphi(T)
            
        phaseflow.ContinuousFunction.__init__(self, function = P, derivative_function = dP)
            
            
class IdealizedLinearBoussinesqBuoyancy(phaseflow.ContinuousFunction):
    
    def __init__(self, rayleigh_numer, prandtl_number, gravity = [0., -1.], reynolds_number = 1.):
    
        g = fenics.Constant(gravity)
        
        Ra = fenics.Constant(rayleigh_numer)
        
        Pr = fenics.Constant(prandtl_number)
        
        Re = fenics.Constant(reynolds_number)
        
        def f_B(T):
        
            return T*Ra/(Pr*Re**2)*g
    
    
        def df_B(T):
            """In this case the derivative df_B is no longer temperature-dependent,
            but we still define this as a function of temperature so that the 
            interface is consistent.
            """

            return Ra/(Pr*Re**2)*g
    
        phaseflow.ContinuousFunction.__init__(self, function=f_B, derivative_function=df_B)
        
        
class TanhSemiPhasefieldMapping(phaseflow.ContinuousFunction):

    def __init__(self, regularization_smoothing_parameter, regularization_central_temperature = 0.):
    
        T_r = fenics.Constant(regularization_central_temperature)
        
        r = fenics.Constant(regularization_smoothing_parameter)
    
        def phi(T):

            return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
        def sech(theta):
        
            return 1./fenics.cosh(theta)
    
    
        def dphi(T):
        
            return -sech((T_r - T)/r)**2/(2.*r)
            
        phaseflow.ContinuousFunction.__init__(self, function=phi, derivative_function=dphi)
        

class ConstantFunction(phaseflow.ContinuousFunction):

    def __init__(self, constant_value):
    
        constant = fenics.Constant(constant_value)
        
        def f(T):
    
            return constant
        
        def df(T):
        
            return 0.*constant
            
        phaseflow.core.ContinuousFunction.__init__(self, function=f, derivative_function=df)
 