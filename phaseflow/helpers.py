""" **helpers.py** contains a variety of patching code. """
import inspect
import errno    
import os
import fenics


def run_simulation_with_temporary_output(simulation):
    """ This is needed to run the test suite with Travis-CI. 
    
    Otherwise, the process will attempt to write outputs to the working directory,
    resulting in a permission error.
    
    Parameters
    ----------
    simulation : phaseflow.Simulation
    """
    simulation.prefix_output_dir_with_tempdir = True
    
    simulation.run()


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
        
        self.parameters["functions_share_mesh"] = True  # Efficiently handles time-dependent solutions

        self.parameters["flush_output"] = True  # This allows us to view the solution while still running.
        
        self.path = filepath  # Mimic the file path attribute from a `file` returned by `open` 
    

def print_once(message):
    """ Print only if the process has MPI rank 0.
    
    This is called throughout Phaseflow instead of `print` so that MPI runs don't duplicate messages.
    
    Parameters
    ----------
    message : string
    """
    if fenics.dolfin.MPI.rank(fenics.dolfin.mpi_comm_world()) is 0:
    
        print(message)
    
    
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
    