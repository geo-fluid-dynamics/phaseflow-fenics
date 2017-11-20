from builtins import str
from builtins import range
from .context import phaseflow
import fenics
import scipy.optimize as opt

T_f = 0.1

def melt_toy_pcm(output_dir = "output/test_melt_toy_pcm/",
        restart = False, restart_filepath = '', start_time = 0.):
    
    
    # Make the mesh.
    initial_mesh_size = 1
    
    mesh = fenics.UnitSquareMesh(initial_mesh_size, initial_mesh_size, 'crossed')
    
    initial_hot_wall_refinement_cycles = 6
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        edge_markers = fenics.EdgeFunction("bool", mesh)
        
        hot_wall.mark(edge_markers, True)

        fenics.adapt(mesh, edge_markers)
        
        mesh = mesh.child()
    
    
    # Run phaseflow.
    T_hot = 1.
    
    T_cold = -0.1
    
    w, mesh = phaseflow.run(
        stefan_number = 1.,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        mesh = mesh,
        time_step_size = 1.e-3,
        start_time = start_time,
        end_time = 0.02,
        stop_when_steady = True,
        temperature_of_fusion = T_f,
        regularization_smoothing_factor = 0.05,
        adaptive = True,
        adaptive_metric = 'phase_only',
        adaptive_solver_tolerance = 1.e-4,
        nlp_relative_tolerance = 1.e-8,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "("+str(T_hot)+" - "+str(T_cold)+")*(x[0] < 0.001) + "+str(T_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(T_hot), 'degree': 2,
                'location_expression': "near(x[0],  0.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(T_cold), 'degree': 2,
                'location_expression': "near(x[0],  1.)",
                'method': "topological"}],
        output_dir = output_dir,
        restart = restart,
        restart_filepath = restart_filepath)
    
    return w
    
    
def test_melt_toy_pcm__regression():

    w = melt_toy_pcm()
    
    """
    w = melt_toy_pcm(restart = True, restart_filepath = 'output/test_melt_toy_pcm/restart_t0.02.h5',
        start_time = 0.02,
        output_dir = 'output/test_melt_toy_pcm/restart_t0.02/')
    """
    
    
    # Verify against a reference solution.
    pci_y_position_to_check =  0.875
    
    reference_pci_x_position = 0.226
    
    def T_minus_T_f(x):
    
        wval = w.leaf_node()(fenics.Point(x, pci_y_position_to_check))
        
        return wval[3] - T_f

        
    pci_x_position = opt.newton(T_minus_T_f, 0.01)
    
    assert(abs(pci_x_position - reference_pci_x_position) < 1.e-2)
    
    
if __name__=='__main__':

    test_melt_toy_pcm__regression()
    
    