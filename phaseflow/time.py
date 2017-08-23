import fenics
import dolfin
import helpers
import output


TIME_EPS = 1.e-8

class TimeStepSize(helpers.BoundedValue):

    def __init__(self, bounded_value):
    
        super(TimeStepSize, self).__init__(bounded_value.min, bounded_value.value, bounded_value.max)

    
    def set(self, value):
    
        assert(value > TIME_EPS)
        
        old_value = self.value
        
        super(TimeStepSize, self).set(value)
        
        if abs(self.value - old_value) > dolfin.DOLFIN_EPS:
        
            helpers.print_once("Set time step size to dt = "+str(value))


def adaptive_time_step(time_step_size, w, w_n, bcs, solve_time_step, debug=False):
    
    converged = False
    
    while not converged:
    
        converged = solve_time_step(dt=time_step_size.value, w=w, w_n=w_n, bcs=bcs)
        
        if not converged:
        
            if debug:
        
                newton_files = [fenics.XDMFFile('debug/newton_velocity.xdmf'),
                    fenics.XDMFFile('debug/newton_pressure.xdmf'),
                    fenics.XDMFFile('debug/newton_temperature.xdmf')]
        
                output.write_solution(newton_files, w, time=-1.) 
        
            w.assign(w_n)
        
        if time_step_size.value <= time_step_size.min + dolfin.DOLFIN_EPS:
            
            break;
        
        if not converged:
        
            time_step_size.set(time_step_size.value/2.)
    
    return converged
   
   
def check(current_time, time_step_size, final_time, output_times, output_count):

    output_this_time = False
        
    next_time = current_time + time_step_size.value
    
    if next_time > final_time:
    
        next_time = final_time
        
        time_step_size.set(next_time - current_time)
    
    if output_times is not ():
    
        next_output_time = output_times[output_count]
        
        if next_time > next_output_time:
        
            next_time = next_output_time
            
            time_step_size.set(next_time - current_time)
        
        if next_output_time == 'final':
           
            next_output_time = final_time
    
        if output_times[output_count] == 'all':
        
            next_output_time = next_time
            
        if abs(next_time - next_output_time) < TIME_EPS:
    
            output_this_time = True
            
            output_count += 1
            
    return time_step_size, next_time, output_this_time, output_count 
   

def steady(W, w, w_n):
    '''Check if solution has reached an approximately steady state.'''
    
    '''@todo Get function space info from w or w_n rather than passing the first argument'''
    STEADY_RELATIVE_TOLERANCE = 1.e-4 # @todo: Expose this parameter
    
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, 'L2')/fenics.norm(w_n, 'L2')
    
    helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < STEADY_RELATIVE_TOLERANCE):
        
        steady = True
    
    return steady

    
if __name__=='__main__':

    TimeStepSize()