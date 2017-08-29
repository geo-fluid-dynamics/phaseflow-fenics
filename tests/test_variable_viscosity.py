from .context import phaseflow
import fenics


def verify_against_ghia1982(w, mesh):

    data = {'Re': 100, 'x': 0.5, 'y': [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000], 'ux': [1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
            
            ux = wval[0]
            
            assert(abs(ux - true_ux) < 2.e-2)
            

def variable_viscosity(m=20, start_time = 0., end_time = 1000., time_step_bounds = (0.1, 0.1, 10.),
    output_times = ('start', 1., 10., 100., 'end'), mu_s = 1.e6,
    initial_pci_refinement_cycles = 4, theta_s = 0., R_s = 0.05, restart = False):

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    left_middle = 'near(x[0], 0.) && near(x[1], 0.5)'
    
    output_dir = 'output/variable_viscosity_m'+str(m)+'_mus'+str(mu_s)+'_thetas'+str(theta_s)+'_Rs'+str(R_s)
    
    restart_filepath=''
    
    if restart:
    
        restart_filepath = output_dir+'/restart_t'+str(start_time)+'.hdf5'
        
        output_dir = output_dir+'_restart'+str(start_time)
        
    w, mesh = phaseflow.run(
        debug = True,
        restart = restart,
        restart_filepath = restart_filepath,
        automatic_jacobian = False,
        mesh = fenics.RectangleMesh(fenics.Point(0., -0.25), fenics.Point(1., 1.), m, m, 'crossed'),
        start_time = start_time,
        end_time = end_time,
        time_step_bounds = time_step_bounds,
        output_times = output_times,
        stop_when_steady = True,
        steady_relative_tolerance = 1.e-4,
        K = 0.,
        mu_l = 0.01,
        mu_s = mu_s,
        regularization = {'a_s': 2., 'theta_s': theta_s, 'R_s': R_s},
        nlp_relative_tolerance = 1.e-4,
        nlp_max_iterations = 30,
        max_pci_refinement_cycles_per_time = 4,
        initial_pci_refinement_cycles = initial_pci_refinement_cycles,
        g = (0., 0.),
        Ste = 1.e16,
        output_dir = output_dir,
        initial_values_expression = (lid, "0.", "0.", "1. - 2.*(x[1] <= 0.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("1.", "0."), 'degree': 3, 'location_expression': lid, 'method': 'topological'},
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': fixed_walls, 'method': 'topological'},
            {'subspace': 1, 'value_expression': "0.", 'degree': 2, 'location_expression': left_middle, 'method': 'pointwise'}])
            
    verify_against_ghia1982(w, mesh)

            
def test_variable_viscosity():

    variable_viscosity(end_time = 20., time_step_bounds = (0.1, 0.1, 3.), 
        output_times = ('start', 'end'),
        theta_s = -0.01, R_s = 0.01, )
    
    
if __name__=='__main__':

    test_variable_viscosity()
    