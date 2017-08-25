from .context import phaseflow


def test_time_check():

    time_step_size = phaseflow.helpers.BoundedValue(0.5, 1., 1.)
    
    final_time = 3.
    
    output_count = 0
    
    current_time = 0.
    
    ''' Here we set output times such that an additional time point at 0.5 will be added,
        and so output at times t = 2. will be skipped.
        The time step size will automatically be reduced to 0.5'''
    output_times = ('start', 0.5, 1., 'end')
                
    output_this_time = True
    
    output_count += 1 
    
    assert(output_this_time)
    
    assert(output_count == 1)
    
    print(current_time)
    
    
    time_step_size, next_time, output_this_time, output_count = phaseflow.time.check(current_time,
            time_step_size, final_time, output_times, output_count)
            
    assert(output_this_time)
    
    assert(output_count == 2)
    
    assert(time_step_size.value == 0.5)
    
    current_time += time_step_size.value
    
    print(current_time)
    
    
    time_step_size, next_time, output_this_time, output_count = phaseflow.time.check(current_time,
            time_step_size, final_time, output_times, output_count)
            
    assert(output_this_time)
    
    assert(output_count == 3)
    
    current_time += time_step_size.value
    
    print(current_time)
    
    
    time_step_size.set(1.)
    
    time_step_size, next_time, output_this_time, output_count = phaseflow.time.check(current_time,
            time_step_size, final_time, output_times, output_count)

    assert(not output_this_time)
    
    assert(output_count == 3)
    
    current_time += time_step_size.value
    
    print(current_time)
    
    
    time_step_size, next_time, output_this_time, output_count = phaseflow.time.check(current_time,
        time_step_size, final_time, output_times, output_count)
        
    assert(output_this_time)
    
    assert(output_count == 4)
    
    current_time += time_step_size.value
    
    print(current_time)
    

if __name__=='__main__':
    
    test_time_check()
    