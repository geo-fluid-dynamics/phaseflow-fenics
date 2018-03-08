""" **state.py** contains the State class. """
import fenics
import phaseflow


class State:
    """ Contain a time-dependent `solution` which is a `fenics.Function` and an associated time,
    along with associated methods, e.g. for interpolating mathematical expressions onto the solution,
    and writing the solution to a file.
    
    References to the function space and element are saved as attributes, since they are needed for the
    `self.interpolate` method.
    
    Parameters
    ----------
    function_space : fenics.FunctionSpace
    
        This is the function space on which lives the solution.
    
    element : fenics.MixedElement
    
        Ideally the function space should already know the element; but the author has failed to find it.
        So, we store this reference separately.
    """
    def __init__(self, function_space, element):
        """ Set the solution, associated time, and associated function space and element. """
        self.solution = fenics.Function(function_space)
        
        self.time = 0.
        
        self.function_space = function_space
        
        self.element = element
        
    
    def interpolate(self, expression_strings):
        """Interpolate the solution from mathematical expressions.

        Parameters
        ----------
        expression_strings : tuple of strings 
            
            Each string will be an argument to a `fenics.Expression`.
        """
        interpolated_solution = fenics.interpolate(
            fenics.Expression(expression_strings, element = self.element), 
            self.function_space.leaf_node())
        
        self.solution.leaf_node().vector()[:] = interpolated_solution.leaf_node().vector() 
        
    
    def write_solution(self, file):
        """ Write the solution to a file.
        
        This assumes that `self.state.solution` has exactly three parts, including, in order, 
        the pressure, velocity, and temperature. 
        For other types of solutions, e.g. with more variables, this method must be overloaded.
        
        Parameters
        ----------
        file : phaseflow.helpers.SolutionFile
        
            This method should have been called from within the context of the open `file`.
        """
        phaseflow.helpers.print_once("Writing solution to " + str(file.path))
        
        pressure, velocity, temperature = self.solution.leaf_node().split()
    
        pressure.rename("p", "pressure")
        
        velocity.rename("u", "velocity")
        
        temperature.rename("T", "temperature")
        
        for var in [pressure, velocity, temperature]:
        
            file.write(var, self.time)
            