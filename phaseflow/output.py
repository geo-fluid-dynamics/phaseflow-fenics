"""This module contains functions for writing solutions to disk."""
import helpers


def write_solution(solution_file, w, time):
    """Write the solution to disk."""

    helpers.print_once("Writing solution to HDF5+XDMF")
    
    velocity, pressure, temperature = w.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("theta", "temperature")
    
    for i, var in enumerate([velocity, pressure, temperature]):
    
        solution_file.write(var, time)

        
if __name__=='__main__':

    pass
    