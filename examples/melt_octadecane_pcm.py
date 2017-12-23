import fenics
import phaseflow


def melt_octadecane_pcm(output_dir = "output/melt_octadecane_pcm/",
        end_time = 80.,
        restart = False, restart_filepath = '', start_time = 0.):
    

    T_f = 0.1
    
    
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
    
    initial_pci_position = 1./float(initial_mesh_size)/2.**(initial_hot_wall_refinement_cycles - 1)
    
    w, mesh = phaseflow.run(
        stefan_number = 0.045,
        rayleigh_number = 3.27e5,
        prandtl_number = 56.2,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        mesh = mesh,
        time_step_size = 0.1,
        start_time = start_time,
        end_time = end_time,
        stop_when_steady = False,
        temperature_of_fusion = T_f,
        regularization_smoothing_factor = 0.05,
        adaptive = True,
        adaptive_metric = 'phase_only',
        adaptive_solver_tolerance = 1.e-5,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 200,
        nlp_relaxation = 0.45,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "(" + str(T_hot) + " - " + str(T_cold) + ")*(x[0] < " + str(initial_pci_position) + ") + " + str(T_cold)),
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
    
    
if __name__=='__main__':

    melt_octadecane_pcm()
    