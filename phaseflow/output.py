import helpers

def write_solution(solution_files, W, _w, current_time):

    helpers.print_once("Writing solution to HDF5+XDMF")
    
    w = _w.leaf_node()
        
    velocity, pressure, temperature = w.split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("theta", "temperature")
        
    for i, var in enumerate([velocity, pressure, temperature]):
    
        solution_files[i].write(var, current_time)
            