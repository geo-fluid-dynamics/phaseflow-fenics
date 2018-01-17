""" This module contains a variety of useful classes and functions."""
import fenics
import phaseflow


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
    
            
if __name__=="__main__":

    pass
    