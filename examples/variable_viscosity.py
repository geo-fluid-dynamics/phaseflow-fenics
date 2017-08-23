import fenics
import phaseflow

    
def variable_viscosity(m=42, mu_s = 1.e6, theta_s = 0.01, R_s = 0.05, restart = False):

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    left_middle = 'near(x[0], 0.) && near(x[1], 0.5)'

    w, mesh = phaseflow.run(
        debug = True,
        restart = False,
        automatic_jacobian = False,
        mesh = fenics.RectangleMesh(fenics.Point(0., -0.25), fenics.Point(1., 1.), m, m, 'crossed'),
        final_time = 1000.,
        time_step_bounds = (0.1, 0.1, 10.),
        output_times = ('initial', 0.1, 1., 10., 100., 'final'),
        stop_when_steady = True,
        steady_relative_tolerance = 1.e-4,
        K = 0.,
        mu_l = 0.01,
        mu_s = mu_s,
        regularization = {'a_s': 2., 'theta_s': theta_s, 'R_s': R_s},
        nlp_relative_tolerance = 1.e-4,
        nlp_max_iterations = 20,
        max_pci_refinement_cycles = 4,
        g = (0., 0.),
        Ste = 1.e16,
        output_dir='output/variable_viscosity_m'+str(m)+'_mus'+str(mu_s)+'_thetas'+str(theta_s)+'_Rs'+str(R_s),
        initial_values_expression = (lid, "0.", "0.", "1. - 2.*(x[1] <= 0.)"),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': ("1.", "0."), 'degree': 3, 'location_expression': lid, 'method': 'topological'},
            {'subspace': 0, 'value_expression': ("0.", "0."), 'degree': 3, 'location_expression': fixed_walls, 'method': 'topological'},
            {'subspace': 1, 'value_expression': "0.", 'degree': 2, 'location_expression': left_middle, 'method': 'pointwise'}])

            
def run_variable_viscosity():

    variable_viscosity(m=20, mu_s=1.e6, theta_s = 0., R_s=0.1, restart=False)
    
    
if __name__=='__main__':

    run_variable_viscosity()