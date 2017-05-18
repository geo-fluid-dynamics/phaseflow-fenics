import phaseflow

from fenics import UnitSquareMesh, Point


''' @todo Is the time from danaila2014newton Fig 7 t = 71. or t = 78.7? This appears contradictory in the article'''

def test_melt_pcm_linearized():

    coarse_m = 10
    
    theta_hot = 1.
    
    theta_cold = -0.01    
    
    Re = phaseflow.REYNOLDS_NUMBER
    
    Ra = 3.27e5
    
    Pr = 56.2
    
    Ste = 0.045
    
    theta0 = theta_cold
    
    w = phaseflow.run(
        output_dir = "output/test_melt_pcm_linearized",
        linearize = True,
        Ra = Ra,
        Pr = Pr,
        Ste = Ste,
        m_B = lambda theta : theta*Ra/(Pr*Re**2),
        dm_B_dtheta = lambda theta : Ra/(Pr*Re**2),
        mesh = UnitSquareMesh(coarse_m, coarse_m, "crossed"),
        adaptive_space = False,
        adaptive_space_error_tolerance = 1.e-4,
        time_step_size = phaseflow.BoundedValue(1.e-5, 1.e-5, 1.e-5),
        final_time = 78.7,
        newton_relative_tolerance = 1.e-4,
        max_newton_iterations = 10,
        theta0 = theta0,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            str(theta0)+" + near(x[0],  0.)*("+str(theta_hot)+" - "+str(theta_cold)+")"),
        bc_expressions = [
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"],
        [2, "0.", 2, "near(x[0],  0.)", "topological"],
        [2, "0.", 2, "near(x[0],  1.)", "topological"]])
        
        
if __name__=='__main__':

    test_melt_pcm_linearized()