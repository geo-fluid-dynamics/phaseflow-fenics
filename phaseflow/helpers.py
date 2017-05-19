import inspect


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