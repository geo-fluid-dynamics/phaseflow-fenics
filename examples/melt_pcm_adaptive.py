import fenics
import phaseflow
        
def melt_pcm(
        dt = 1.e-3,
        output_dir='output/melt_pcm_adaptive',
        start_time=0.,
        end_time=0.05,
        initial_hot_wall_refinement_cycles = 6,
        nlp_max_iterations = 30,
        initial_mesh_size = 1,
        restart=False,
        restart_filepath=''):

        
    #
    mesh = fenics.UnitSquareMesh(initial_mesh_size, initial_mesh_size, 'crossed')
    
    class HotWall(fenics.SubDomain):
        
        def inside(self, x, on_boundary):
        
            return on_boundary and fenics.near(x[0], 0.)

            
    hot_wall = HotWall()
    
    for i in range(initial_hot_wall_refinement_cycles):
        
        edge_markers = fenics.EdgeFunction("bool", mesh)
        
        hot_wall.mark(edge_markers, True)

        fenics.adapt(mesh, edge_markers)
        
        mesh = mesh.child()
    
    
    #
    theta_hot = 1.
    
    theta_cold = -0.1

    w, mesh = phaseflow.run(
        Ste = 1.,
        Ra = 1.,
        Pr = 1.,
        mu_s = 1.e4,
        mu_l = 1.,
        mesh = mesh,
        time_step_size = dt,
        start_time = start_time,
        end_time = end_time,
        stop_when_steady = True,
        regularization = {'T_f': 0.1, 'r': 0.05},
        nlp_max_iterations = nlp_max_iterations,
        nlp_relative_tolerance = 1.e-4,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            "("+str(theta_hot)+" - "+str(theta_cold)+")*(x[0] < 0.001) + "+str(theta_cold)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_hot), 'degree': 2,
                'location_expression': "near(x[0],  0.)",
                'method': "topological"},
            {'subspace': 2, 'value_expression': str(theta_cold), 'degree': 2,
                'location_expression': "near(x[0],  1.)",
                'method': "topological"}],
        output_dir = output_dir,
        debug = True,
        restart = restart,
        restart_filepath = restart_filepath)

    return w, mesh
    
    
if __name__=='__main__':

    melt_pcm(end_time=0.02)
    