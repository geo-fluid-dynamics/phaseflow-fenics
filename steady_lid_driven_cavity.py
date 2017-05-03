import nsb_pcm as ns

from fenics import UnitSquareMesh


def run(linearize = False, Re = 1., m=8):

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'

    ns.run(linearize = linearize, \
        adaptive_time = False, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1.e12, \
        time_step_size = 1.e12, \
        mu = 1./Re, \
        output_dir="output/steady_lid_driven_cavity_Re"+str(Re)+"_m"+str(m), \
        s_theta ='0.', \
        initial_values_expression = ('0.', '0.', '0.', '0.'), \
        bc_expressions = [[0, ('1.', '0.'), 3, lid], [0, ('0.', '0.'), 3, fixed_walls], [1, '0.', 2, bottom_left_corner]])
        
        
def test():

    run()
    
    pass
    
    
if __name__=='__main__':

    test()