import fenics
import phaseflow
        
def melt_pcm_3d(
        initial_mesh_size = [10, 10, 1],
        dt = 1.e-3,
        initial_hot_wall_refinement_cycles = 2,
        output_dir='output/melt_pcm_3d',
        start_time=0.,
        end_time=0.05,
        nlp_max_iterations = 50,
        nlp_relaxation = 1.,
        restart=False,
        restart_filepath=''):

    # Make the mesh.
    mesh = fenics.BoxMesh(fenics.Point(0., 0., -0.2),
            fenics.Point(1., 1., 0.2),
            initial_mesh_size[0], initial_mesh_size[1], initial_mesh_size[2])
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        cell_markers = fenics.CellFunction("bool", mesh, False)
        
        for cell in fenics.cells(mesh):
            
            found_left_boundary = False
            
            for vertex in fenics.vertices(cell):
                
                if fenics.near(vertex.x(0), 0.):
                    
                    found_left_boundary = True
                    
                    break
                    
            if found_left_boundary:
                
                cell_markers[cell] = True
                
        mesh = fenics.refine(mesh, cell_markers)
        
        
    #
    T_hot = 1.
    
    T_cold = -0.1
    
    initial_pci_position = 1./float(initial_mesh_size[0])/2.**(initial_hot_wall_refinement_cycles - 1)

    w, mesh = phaseflow.run(
        stefan_number = 1.,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        solid_viscosity = 1.e4,
        liquid_viscosity = 1.,
        gravity = (0., -1., 0.),
        mesh = mesh,
        time_step_size = dt,
        start_time = start_time,
        end_time = end_time,
        temperature_of_fusion = 0.1,
        regularization_smoothing_factor = 0.025,
        adaptive = True,
        adaptive_metric = 'phase_only',
        adaptive_solver_tolerance = 1.e-4,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = nlp_max_iterations,
        nlp_relaxation = nlp_relaxation,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "0.",
            "("+str(T_hot)+" - "+str(T_cold)+")*(x[0] < 0.001) + "+str(T_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0.", "0."), 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.) | near(x[2], -0.2) | near(x[2], 0.2)",
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

    return w, mesh
    
    
def run_melt_pcm_3d():
    
    w, mesh = melt_pcm_3d(output_dir = "output/melt_pcm_3d/", end_time = 0.02, nlp_max_iterations = 200,
        restart = False,
        restart_filepath = "",
        start_time = 0.,
        nlp_relaxation = 0.8)
    
if __name__=='__main__':

    run_melt_pcm_3d()
