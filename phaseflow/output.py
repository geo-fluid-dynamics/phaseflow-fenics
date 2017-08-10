import helpers

def write_solution(output_format, solution_files, W, _w, current_time):

    helpers.print_once("Writing solution to "+output_format)
    
    w = _w.leaf_node()
        
    velocity, pressure, temperature = w.split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("theta", "temperature")

    if (output_format is 'vtk') or (output_format is 'xdmf'):
        
        for i, var in enumerate([velocity, pressure, temperature]):
        
            solution_files[i].write(var, current_time)
            
    elif output_format is 'table':
    
        ''' The following table output is just hacked on for the 1D Stefan Problem, not general at all.'''
    
        coordinates = W.tabulate_dof_coordinates()
        
        for i, x in enumerate(coordinates):
        
            solution_files[0].write(str(current_time)+", "+str(x)+", "+str(temperature(x))+"\n")