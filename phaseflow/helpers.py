""" **helpers.py** contains a variety of patching code. """
import pathlib
import fenics


class Point(fenics.Point):
    """ This class extends `fenics.Point` with a more convenient constructor for 1D/2D/3D. 
    
    Parameters
    ----------
    coordinates : tuple of floats
    """
    def __init__(self, coordinates):
    
        if type(coordinates) is type(0.):
        
            coordinates = (coordinates,)
        
        if len(coordinates) == 1:
        
            fenics.Point.__init__(self, coordinates[0])
            
        elif len(coordinates) == 2:
        
            fenics.Point.__init__(self, coordinates[0], coordinates[1])
            
        elif len(coordinates) == 3:
        
            fenics.Point.__init__(self, coordinates[0], coordinates[1], coordinates[2])

            
class SolutionFile(fenics.XDMFFile):
    """ This class extends `fenics.XDMFFile` with some minor changes for convenience. 
    
    Parameters
    ----------
    filepath : string
    """
    def __init__(self, filepath):

        fenics.XDMFFile.__init__(self, filepath)
        
        self.parameters["functions_share_mesh"] = True  # This refers to the component solution functions.

        self.parameters["flush_output"] = True  # This allows us to view the solution while still running.
        
        self.path = filepath  # Mimic the file path attribute from a `file` returned by `open` 


def mkdir_p(pathstring):
    """ Make a directory if it doesn't exist.
    
    This is needed because `open` does not create directories.
    
    Now this just calls the appropriate function from pathlib.
    Older versions were more complicated.
    
    Parameters
    ----------
    path : string
    """
    path = pathlib.Path(pathstring)
    
    path.mkdir(parents = True, exist_ok = True)

    
def float_in(float_item, float_collection, tolerance = 1.e-8):
    
    for item in float_collection:
    
        if abs(float_item - item) < tolerance:
        
            return True
    
    return False
    
    
if __name__=="__main__":

    pass
    