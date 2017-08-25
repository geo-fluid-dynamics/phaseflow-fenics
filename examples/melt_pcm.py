import fenics
import phaseflow
        
def melt_pcm(Ste = 0.045,
        Ra = 3.27e5,
        Pr = 56.2,
        mu_s = 1.e8,
        epsilon_1 = 0.01,
        m = 10,
        time_step_bounds = (1.e-3, 1.e-3, 1.e-3),
        initial_pci_refinement_cycles = 6,
        max_pci_refinement_cycles = 6,
        output_dir='output/melt_pcm',
        start_time=0.,
        end_time=1.,
        nlp_divergence_threshold = 1.e12,
        nlp_relaxation = 1.,
        restart=False,
        restart_filepath=''):

    theta_hot = 1.
    
    theta_cold = -0.1

    w, mesh = phaseflow.run(
        Ste = Ste,
        Ra = Ra,
        Pr = Pr,
        mu_s = mu_s,
        mu_l = 1.,
        mesh = fenics.UnitSquareMesh(m, m),
        time_step_bounds = time_step_bounds,
        start_time = start_time,
        end_time = end_time,
        output_times = ('all',),
        stop_when_steady = True,
        automatic_jacobian = False,
        custom_newton = True,
        regularization = {'a_s': 2., 'theta_s': epsilon_1, 'R_s': epsilon_1/2.},
        initial_pci_refinement_cycles = initial_pci_refinement_cycles,
        max_pci_refinement_cycles = max_pci_refinement_cycles,
        nlp_max_iterations = 30,
        nlp_divergence_threshold = nlp_divergence_threshold,
        nlp_relaxation = nlp_relaxation,
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
    
    w, mesh = melt_pcm(Ste = 1.,
        Ra = 1.,
        Pr = 1.,
        mu_s = 1.e4,
        epsilon_1 = 0.1,
        m = 10,
        time_step_bounds = (1.e-4, 1.e-3, 1.e-2),
        initial_pci_refinement_cycles = 4,
        max_pci_refinement_cycles = 6,
        nlp_divergence_threshold = 1.e12,
        nlp_relaxation = 0.5,
        output_dir = 'output/melt_pcm_0')
    
    w, mesh = melt_pcm(Ste = 1.,
        Ra = 1.,
        Pr = 1.,
        mu_s = 1.e4,
        epsilon_1 = 0.1,
        time_step_bounds = 1.e-3,
        initial_pci_refinement_cycles = 1,
        max_pci_refinement_cycles = 4,
        nlp_divergence_threshold = 1.e12,
        nlp_relaxation = 0.5,
        restart = True,
        restart_filepath = 'output/melt_pcm_0/restart_t0.012.hdf5',
        start_time = 0.012,
        output_dir = 'output/melt_pcm_1')

 
    
if __name__=='__main__':

    run_melt_pcm()
