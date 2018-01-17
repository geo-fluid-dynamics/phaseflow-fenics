import fenics


class DanailaTaylorHoodElement(fenics.MixedElement):

    def __init__(self, pressure_degree = 1, temperature_degree = 1, cell):
        
        pressure_element = fenics.FiniteElement("P", cell, pressure_degree)
        
        velocity_degree = pressure_degree + 1
        
        velocity_element = fenics.VectorElement("P", cell, velocity_degree)

        temperature_element = fenics.FiniteElement("P", cell, temperature_degree)

        fenics.MixedElement.__init__(self, [pressure_element, velocity_element, temperature_element])


class PhaseDependentMaterialProperty(ContinuousFunction):

    def __init__(self, semi_phasefield_mapping, liquid_value, solid_value):
    
        P_L = fenics.Constant(liquid_value)
        
        P_S = fenics.Constant(solid_value)
        
        phi = semi_phasefield_mapping.function
        
        def P(T):
        
            return P_L + (P_S - P_L)*phi(T)
        

        dphi = semi_phasefield_mapping.derivative_function
        
        def dP(T):
        
            return (P_S - P_L)*dphi(T)
            
            
class IdealizedLinearBoussinesqBuoyancy(ContinuousFunction)
    
    def __init__(self, gravity = [0., -1.], reynolds_number = 1., rayleigh_numer, prandtl_number):
    
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
    
        ContinuousFunction.__init__(self, function=f_B, derivative_function=df_B)
        
        
class TanhSemiPhasefieldMapping(ContinuousFunction):

    def __init__(self, regularization_central_temperature = 0., regularization_smoothing_parameter)
    
        T_r = fenics.Constant(regularization_central_temperature)
        
        r = fenics.Constant(regularization_smoothing_parameter)
    
        def phi(T):

            return 0.5*(1. + fenics.tanh((T_r - T)/r))
        
        
        def sech(theta):
        
            return 1./fenics.cosh(theta)
    
    
        def dphi(T):
        
            return -sech((T_r - T)/r)**2/(2.*r)
            
        ContinuousFunction.__init__(self, function=phi, derivative_function=dphi)
        
       
class ConstantPhase(ContinuousFunctionOfTemperature):

    def __init__(self, constant_value)
    
        constant = fenics.Constant(constant_value)
        
        def phi(T):
    
            return constant
        
        def dphi(T):
        
            return 0.
            
        ContinuousFunctionOfTemperature.__init__(self, function=phi, derivative_function=dphi)
        
        
def write_solution(solution_file, solution, time, solution_filepath):
    """Write the solution to disk.
    
    Parameters
    ----------
    solution_file : fenics.XDMFFile
    
        write_solution should have been called from within the context of an open fenics.XDMFFile.
    
    solution : fenics.Function
    
        The FEniCS function where the solution is stored.
    
    time : float
    
        The time corresponding to the time-dependent solution.
    
    solution_filepath : str
    
        This is needed because fenics.XDMFFile does not appear to have a method for providing the file path.
        With a Python file, one can simply do
        
            File = open("foo.txt", "w")
            
            File.name
            
        But fenics.XDMFFile.name returns a reference to something done with SWIG.
    
    """
    phaseflow.helpers.print_once("Writing solution to " + str(solution_filepath))
    
    pressure, velocity, temperature = solution.leaf_node().split()
    
    pressure.rename("p", "pressure")
    
    velocity.rename("u", "velocity")
    
    temperature.rename("T", "temperature")
    
    for i, var in enumerate([pressure, velocity, temperature]):
    
        solution_file.write(var, time)
        
        