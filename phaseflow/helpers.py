""" **helpers.py** contains a variety of patching code. """
import inspect
import errno    
import os
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

            
def mkdir_p(path):
    """ Make a directory if it doesn't exist.
    
    This is needed because `open` does not create directories.
    
    Code from https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python 
    
    Parameters
    ----------
    path : string
    """
    try:
    
        os.makedirs(path)
        
    except OSError as exc:  # Python >2.5
    
        if exc.errno == errno.EEXIST and os.path.isdir(path):
        
            pass
            
        else:
        
            raise

            
if __name__=="__main__":

    pass
    