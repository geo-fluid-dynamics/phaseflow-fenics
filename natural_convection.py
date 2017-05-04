import nsb_pcm as ns

from fenics import UnitSquareMesh


def run():

    ns.run( \
        output_dir = "output/natural_convection_nonlinear", \
        linearize = False, adaptive_time = False, \
        bc_expressions = [ \
        [0, ("0.", "0."), 3, "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)","topological"], \
        [2, "0.5", 2, "near(x[0],  0.)", "topological"], \
        [2, "-0.5", 2, "near(x[0],  1.)", "topological"]])
        
        
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()