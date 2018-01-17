""" This module contains a variety of useful classes and functions."""
import fenics
import phaseflow


TIME_EPS = 1.e-8

MAX_TIME_STEPS = 1000000000000

def write_checkpoint(checkpoint_filepath, w, time):
    """Write checkpoint file (with solution and time) to disk."""
    phaseflow.helpers.print_once("Writing checkpoint file to " + checkpoint_filepath)
    
    with fenics.HDF5File(fenics.mpi_comm_world(), checkpoint_filepath, "w") as h5:
                
        h5.write(w.function_space().mesh().leaf_node(), "mesh")
    
        h5.write(w.leaf_node(), "w")
        
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
    
        with h5py.File(checkpoint_filepath, "r+") as h5:
            
            h5.create_dataset("t", data=time)
        
        
def read_checkpoint(checkpoint_filepath):
    """Read the checkpoint solution and time, perhaps to restart."""

    mesh = fenics.Mesh()
        
    with fenics.HDF5File(mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
    
        h5.read(mesh, "mesh", True)
    
    W_ele = make_mixed_fe(mesh.ufl_cell())

    W = fenics.FunctionSpace(mesh, W_ele)

    w = fenics.Function(W)

    with fenics.HDF5File(mesh.mpi_comm(), checkpoint_filepath, "r") as h5:
    
        h5.read(w, "w")
        
    with h5py.File(checkpoint_filepath, "r") as h5:
            
        time = h5["t"].value
        
    return w, time
    

def steady(W, w, w_n, steady_relative_tolerance):
    """Check if solution has reached an approximately steady state."""
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, "L2")/fenics.norm(w_n, "L2")
    
    phaseflow.helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < steady_relative_tolerance):
        
        steady = True
    
    return steady
    
    
if __name__=="__main__":

    pass
    