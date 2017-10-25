import fenics
import phaseflow
import scipy.optimize as opt
        
def verify_pci_position_regression(theta_f, w):

    ref_data = {'Ra': 1., 'Pr': 1., 'theta_f': 0.1, 'r': 0.05, 'Ste': 1., 
        'mu_s': 1.e4,
        'nlp_absolute_tolerance': 1., 'time': 0.001, 'pci_y_pos': 0.5, 'pci_x_pos': 0.043}

    assert(theta_f == ref_data['theta_f'])
            
    def theta_minus_theta_f(x):
    
        wval = w(fenics.Point(x, ref_data['pci_y_pos']))
        
        return wval[3] - theta_f
    
    pci_x_pos = opt.newton(theta_minus_theta_f, 0.01)
    
    assert(abs(pci_x_pos - ref_data['pci_x_pos']) < 1.e-3)
        
        
def melt_pcm(Ste = 0.045,
        Ra = 3.27e5,
        Pr = 56.2,
        mu_s = 1.e8,
        T_f = 0.01,
        r = 0.005,
        initial_mesh_size = 1,
        time_step_size = 1.e-3,
        initial_hot_wall_refinement_cycles = 6,
        output_dir='output/melt_pcm',
        start_time=0.,
        end_time=1.,
        nlp_max_iterations = 30,
        restart=False,
        restart_filepath=''):

    theta_hot = 1.
    
    theta_cold = -0.1
    
    mesh = fenics.UnitSquareMesh(initial_mesh_size, initial_mesh_size)
    
    
    # Refine the initial mesh near the hot wall
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
    w, mesh = phaseflow.run(
        Ste = Ste,
        Ra = Ra,
        Pr = Pr,
        mu_s = mu_s,
        mu_l = 1.,
        mesh = mesh,
        time_step_size = time_step_bounds,
        start_time = start_time,
        end_time = end_time,
        output_times = ('all',),
        stop_when_steady = True,
        regularization = {'T_f': T_f, 'r': r},
        nlp_max_iterations = nlp_max_iterations,
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
    
    
def test_melt_pcm():

    T_f = 0.1
    
    w, mesh = melt_pcm(Ste = 1.,
        Ra = 1.,
        Pr = 1.,
        mu_s = 1.e4,
        T_f = T_f,
        r = T_f/2.,
        time_step_size = 1.e-3,
        end_time = 1.e-3,
        initial_hot_wall_refinement_cycles = 6,
        output_dir = 'output/test_melt_pcm')
        
    verify_pci_position_regression(T_f, w)
    
    
if __name__=='__main__':

    test_melt_pcm()
