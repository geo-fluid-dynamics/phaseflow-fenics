import phaseflow

from fenics import UnitSquareMesh, Point


''' @todo Is the time from danaila2014newton Fig 7 t = 71. or t = 78.7? This appears contradictory in the article'''

def test_melt_pcm():

    m = 4
    
    theta_hot = 1.
    
    theta_cold = -0.01    
    
    Re = phaseflow.Re
    
    Ra = 3.27e5
    
    Pr = 56.2
    
    Ste = 0.045
    
    w = phaseflow.run(
        output_dir = "output/test_melt_pcm",
        linearize = True,
        Ra = Ra,
        Pr = Pr,
        Ste = Ste,
        m_B = lambda theta : theta*Ra/(Pr*Re**2),
        dm_B_dtheta = lambda theta : Ra/(Pr*Re**2),
        mesh = UnitSquareMesh(m, m, "crossed"),
        adaptive_space = True,
        time_step_size = phaseflow.BoundedValue(1.e-3, 1.e-3, 10.),
        final_time = 78.7,
        newton_relative_tolerance = 1.e-4,
        max_newton_iterations = 10,
        initial_values_expression = (
            "0.",
            "0.",
            "0.",
            str(theta_cold)+" + near(x[0],  0.)*("+str(theta_hot)+" - "+str(theta_cold)+")"),
        bc_expressions = [
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"],
        [2, "0.", 2, "near(x[0],  0.)", "topological"],
        [2, "0.", 2, "near(x[0],  1.)", "topological"]])
        
        
if __name__=='__main__':

    test_melt_pcm()