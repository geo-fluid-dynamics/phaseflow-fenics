import helpers
import dolfin

class TimeStepSize(helpers.BoundedValue):

    def __init__(self, bounded_value):
    
        super(TimeStepSize, self).__init__(bounded_value.min, bounded_value.value, bounded_value.max)

        
    def set(self, value):
    
        old_value = self.value
        
        super(TimeStepSize, self).set(value)
        
        if abs(self.value - old_value) > dolfin.DOLFIN_EPS:
        
            print 'Set time step size to dt = ' + str(value)