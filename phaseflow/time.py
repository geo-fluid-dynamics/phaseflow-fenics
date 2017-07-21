import fenics
import dolfin
import helpers


class TimeStepSize(helpers.BoundedValue):

    def __init__(self, bounded_value):
    
        super(TimeStepSize, self).__init__(bounded_value.min, bounded_value.value, bounded_value.max)

    
    def set(self, value):
    
        old_value = self.value
        
        super(TimeStepSize, self).set(value)
        
        if abs(self.value - old_value) > dolfin.DOLFIN_EPS:
        
            print 'Set time step size to dt = ' + str(value)


def adaptive_time_step(time_step_size, w, w_n, bcs, current_time, solve_time_step):
    
    converged = False
    
    while not converged:
    
        converged = solve_time_step(dt=time_step_size.value, w=w, w_n=w_n, bcs=bcs)
        
        if time_step_size.value <= time_step_size.min + dolfin.DOLFIN_EPS:
            
            break;
        
        if not converged:
        
            time_step_size.set(time_step_size.value/2.)
    
    return current_time, converged
   

def steady(W, w, w_n):
    '''Check if solution has reached an approximately steady state.'''
    
    '''@todo Get function space info from w or w_n rather than passing the first argument'''
    STEADY_RELATIVE_TOLERANCE = 1.e-4 # @todo: Expose this parameter
    
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, 'L2')/fenics.norm(w_n, 'L2')
    
    print 'Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = ' + str(unsteadiness)

    if (unsteadiness < STEADY_RELATIVE_TOLERANCE):
        
        steady = True
    
    return steady

    
if __name__=='__main__':

    TimeStepSize()