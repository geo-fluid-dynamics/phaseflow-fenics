import helpers

def write_solution(solution_files, _w, current_time):

    w = _w.leaf_node()
        
    velocity, pressure, temperature = w.split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("theta", "temperature")
        
    for i, var in enumerate([velocity, pressure, temperature]):
    
        solution_files[i] << (var, current_time)
    