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