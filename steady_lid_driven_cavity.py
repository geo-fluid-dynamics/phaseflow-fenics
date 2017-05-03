import nsb_pcm as ns

from fenics import UnitSquareMesh


def run(linearize = False, Re = 1., mu = 1., v = 1., m=32):

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'

    ns.run(linearize = linearize, \
        adaptive_time = False, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1.e12, \
        time_step_size = 1.e12, \
        mu = mu, \
        output_dir="output/steady_lid_driven_cavity_mu"+str(mu)+"_v"+str(v)+"_m"+str(m), \
        s_theta ='0.', \
        initial_values_expression = ('0.', '0.', '0.', '0.'), \
        bc_expressions = [[0, (str(v), '0.'), 3, lid, "topological"], [0, ('0.', '0.'), 3, fixed_walls, "topological"], [1, '0.', 2, bottom_left_corner, "pointwise"]])
        
        
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()