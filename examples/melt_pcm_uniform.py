import fenics
import phaseflow
        
def melt_pcm(
        m = 20,
        dt = 1.e-3,
        output_dir='output/melt_pcm_uniform',
        start_time=0.,
        end_time=0.05,
        nlp_divergence_threshold = 1.e12,
        nlp_max_iterations = 100,
        restart=False,
        restart_filepath=''):

    theta_hot = 1.
    
    theta_cold = -0.1

    w, mesh = phaseflow.run(
        Ste = 1.,
        Ra = 1.,
        Pr = 1.,
        mu_s = 1.e4,
        mu_l = 1.,
        mesh = fenics.UnitSquareMesh(m, m),
        time_step_bounds = dt,
        start_time = start_time,
        end_time = end_time,
        stop_when_steady = True,
        regularization = {'T_f': 0.1, 'r': 0.05},
        custom_newton = True,
        nlp_max_iterations = nlp_max_iterations,
        nlp_divergence_threshold = nlp_divergence_threshold,
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
    
    
def run_melt_pcm(m=20, output_dir='output/melt_pcm_uniform'):
    
    w, mesh = melt_pcm(m = m, output_dir = output_dir)
    
    
if __name__=='__main__':

    for m in [40, 80]:
    
        run_melt_pcm(m=m, output_dir = 'output/melt_pcm_uniform'+str(m)+'/')
