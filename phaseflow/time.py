import fenics
import dolfin
import helpers
import bounded_value
import output


TIME_EPS = 1.e-8

class TimeStepSize(bounded_value.BoundedValue):
    """This class sets bounds on the adaptive time step size."""
    def __init__(self, bounded_value):
    
        super(TimeStepSize, self).__init__(bounded_value.min, bounded_value.value, bounded_value.max)

    
    def set(self, value):
    
        assert(value > TIME_EPS)

        old_value = self.value
        
        super(TimeStepSize, self).set(value)
        
        if abs(self.value - old_value) > dolfin.DOLFIN_EPS:
        
            helpers.print_once("Set time step size to dt = "+str(value))


def adaptive_time_step(time_step_size, w, w_n, bcs, solve_time_step, debug=False):
    """Solve one time step with an adaptive time step size."""
    converged = False
    
    while not converged:
    
        converged = solve_time_step(dt=time_step_size.value, w=w, w_n=w_n, bcs=bcs)
        
        if not converged:
        
            if debug:
        
                with fenics.XDMFFile('debug/newton_solution.xdmf') as newton_file:
        
                    output.write_solution(newton_file, w, time=-1.) 
        
            w.assign(w_n)
        
        if time_step_size.value <= time_step_size.min + dolfin.DOLFIN_EPS:
            
            break;
        
        if not converged:
        
            time_step_size.set(time_step_size.value/2.)
    
    return converged
   
   
def check(current_time, time_step_size, end_time, output_times, output_count):
    """Check if outputs should be written at this time.
    
    This also returns a modified time step size that assures we will
    compute a solution at a specified output time.
    """
    
    """@todo This is too complicated,
    especially because it's doing two things.
    One function should do one thing!
    """
    
    output_this_time = False
        
    next_time = current_time + time_step_size.value

    if next_time > end_time:
    
        next_time = end_time
        
        time_step_size.set(next_time - current_time)
    
    if output_times is not ():
    
        next_output_time = output_times[output_count]
        
        if next_output_time == 'end':
               
            next_output_time = end_time
            
        if next_output_time == 'all':
        
            output_this_time = True
        
        else:
        
            if next_time > next_output_time:
                
                next_time = next_output_time
                
                time_step_size.set(next_time - current_time)
            
            if abs(next_time - next_output_time) < TIME_EPS:
        
                output_this_time = True
                
                if not (output_times[output_count] == 'end'):
                
                    output_count += 1
            
    return time_step_size, next_time, output_this_time, output_count, next_output_time
   

def steady(W, w, w_n, steady_relative_tolerance=1.e-4):
    '''Check if solution has reached an approximately steady state.'''
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, 'L2')/fenics.norm(w_n, 'L2')
    
    helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < steady_relative_tolerance):
        
        steady = True
    
    return steady

    
if __name__=='__main__':

    TimeStepSize()
