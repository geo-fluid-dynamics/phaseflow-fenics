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

    
if __name__=="__main__":

    pass
    