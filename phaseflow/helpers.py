import inspect
import errno    
import os
import fenics

class BoundedValue(object):

    def __init__(self, min=0., value=0., max=0.):
    
        self.min = min
        
        self.value = value
        
        self.max = max
        
    
    def set(self, value):
    
        if value > self.max:
        
            value = self.max
            
        elif value < self.min:
        
            value = self.min
            
        self.value = value
        

def arguments():
    """Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments."""
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    
    posargs = args.pop(posname, [])
    
    args.update(args.pop(kwname, []))
    
    return args, posargs
    

def print_once(string):

    if fenics.dolfin.MPI.rank(fenics.dolfin.mpi_comm_world()) is 0:
    
        print(string)
    
    
''' Make a directory if it doesn't exist.
Code from https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python '''
def mkdir_p(path):

    try:
    
        os.makedirs(path)
        
    except OSError as exc:  # Python >2.5
    
        if exc.errno == errno.EEXIST and os.path.isdir(path):
        
            pass
            
        else:
        
            raise
            