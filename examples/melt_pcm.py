import fenics
import phaseflow
        
def melt_pcm(
        m = 10,
        dt = 1.e-3,
        initial_pci_refinement_cycles = 2,
        max_pci_refinement_cycles_per_time = 6,
        output_dir='output/melt_pcm',
        start_time=0.,
        end_time=0.05,
        nlp_divergence_threshold = 1.e12,
        nlp_max_iterations = 30,
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
        regularization = {'a_s': 2., 'theta_s': 0.1, 'R_s': 0.05},
        initial_pci_refinement_cycles = initial_pci_refinement_cycles,
        max_pci_refinement_cycles_per_time = max_pci_refinement_cycles_per_time,
        minimum_cell_diameter = 0.01,
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
    
    
def run_melt_pcm():
    
    w, mesh = melt_pcm(output_dir = 'output/melt_pcm')
    
    
if __name__=='__main__':

    run_melt_pcm()
