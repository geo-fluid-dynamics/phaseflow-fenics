import nsb_pcm as ns

from fenics import UnitSquareMesh


def run(linearize = False, adaptive_time = False, m=40):

    hot = 0.5
    
    cold = -0.5
    
    if linearize:
    
        hot = cold = 0.

    ns.run( \
        output_dir = "output/natural_convection_linearize"+str(linearize)+"_adaptivetime"+str(adaptive_time)+"_m"+str(m), \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        linearize = linearize, adaptive_time = adaptive_time, \
        bc_expressions = [ \
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"], \
        [2, str(hot), 2, "near(x[0],  0.)", "topological"], \
        [2, str(cold), 2, "near(x[0],  1.)", "topological"]])
        
        
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()