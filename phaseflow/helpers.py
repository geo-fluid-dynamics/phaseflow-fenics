"""**helpers.py** contains a variety of generally useful code.

Containing this code here cleans up the other modules which are more specific to our problem of interest.
"""
import inspect
import errno    
import os
import fenics


def run_simulation_with_temporary_output(simulation):
    """ This is needed to run the test suite with Travis-CI. """
    simulation.prefix_output_dir_with_tempdir = True
    
    simulation.run()


class Point(fenics.Point):

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

    def __init__(self, filepath):

        fenics.XDMFFile.__init__(self, filepath)
        
        self.parameters["functions_share_mesh"] = True  
        
        self.path = filepath
        

def arguments():
    """ Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
    """
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    
    posargs = args.pop(posname, [])
    
    args.update(args.pop(kwname, []))
    
    return args, posargs
    

def print_once(string):

    if fenics.dolfin.MPI.rank(fenics.dolfin.mpi_comm_world()) is 0:
    
        print(string)
    
    

def mkdir_p(path):
    """ Make a directory if it doesn't exist.
    Code from https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python 
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
    