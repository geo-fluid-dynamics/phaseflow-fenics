""" **state.py** contains the State class. """
import fenics
import phaseflow


class State:
    """ Contain a time-dependent `solution` which is a `fenics.Function` and an associated time,
    along with associated methods, e.g. for interpolating mathematical expressions onto the solution,
    and writing the solution to a file.
    
    References to the function space and element are saved as attributes, since they are needed for the
    `self.interpolate` method.
    """
    def __init__(self, function_space, element, time = 0.):
        """
        
        Parameters
        ----------
        function_space : fenics.FunctionSpace
        
        element : fenics.MixedElement
        
        time : float
        """
        self.solution = fenics.Function(function_space)
        
        self.time = time
        
        self.function_space = function_space
        
        self.element = element
        
    
    def interpolate(self, expression_strings):
        """Interpolate the solution from mathematical expressions. """
        interpolated_solution = fenics.interpolate(
            fenics.Expression(expression_strings, element = self.element), 
            self.function_space.leaf_node())
        
        self.solution.leaf_node().vector()[:] = interpolated_solution.leaf_node().vector() 
        
    
    def write_solution(self, file):
        """Write the solution to a file.
        
        Parameters
        ----------
        file : phaseflow.helpers.SolutionFile
        
            write_solution should have been called from within the context of this open file.
        """
        phaseflow.helpers.print_once("Writing solution to " + str(file.path))
        
        pressure, velocity, temperature = self.solution.leaf_node().split()
    
        pressure.rename("p", "pressure")
        
        velocity.rename("u", "velocity")
        
        temperature.rename("T", "temperature")
        
        for var in [pressure, velocity, temperature]:
        
            file.write(var, self.time)
            