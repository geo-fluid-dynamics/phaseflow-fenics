import phaseflow

from fenics import UnitSquareMesh

    
def run(linearize = True, adaptive_time = True, m=20, time_step_size = 1.e-3):

    hot = 0.5
    
    cold = -0.5
    
    if linearize:
    
        hot = cold = 0.

    phaseflow.run( \
        output_dir = "output/natural_convection_linearize"+str(linearize)+"_adaptivetime"+str(adaptive_time)+"_m"+str(m), \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        time_step_size = time_step_size, \
        final_time = 10., \
        stop_when_steady = True, \
        linearize = linearize,
        adaptive_time = adaptive_time, \
        bc_expressions = [ \
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"], \
        [2, str(hot), 2, "near(x[0],  0.)", "topological"], \
        [2, str(cold), 2, "near(x[0],  1.)", "topological"]])

        
if __name__=='__main__':

    run()