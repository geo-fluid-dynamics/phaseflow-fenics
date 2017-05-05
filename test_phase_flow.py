import phase_flow

from fenics import UnitSquareMesh


def default():

    phase_flow.run()


def steady_lid_driven_cavity(linearize = False, mu = 1., v = 1., m=32):

    lid = 'near(x[1],  1.)'

    fixed_walls = 'near(x[0],  0.) | near(x[0],  1.) | near(x[1],  0.)'

    bottom_left_corner = 'near(x[0], 0.) && near(x[1], 0.)'

    phase_flow.run(linearize = linearize, \
        adaptive_time = False, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1.e12, \
        time_step_size = 1.e12, \
        mu = mu, \
        output_dir="output/steady_lid_driven_cavity_mu"+str(mu)+"_v"+str(v)+"_m"+str(m), \
        s_theta ='0.', \
        initial_values_expression = ('0.', '0.', '0.', '0.'), \
        bc_expressions = [[0, (str(v), '0.'), 3, lid, "topological"], [0, ('0.', '0.'), 3, fixed_walls, "topological"], [1, '0.', 2, bottom_left_corner, "pointwise"]])

        
def heat(m=16):

    hot_wall = 'near(x[0], 0.)'
    
    cold_wall = 'near(x[0], 1.)'

    adiabatic_walls = 'near(x[1],  0.) | near(x[1], 1.)'

    phase_flow.run(linearize = True, \
        adaptive_time = True, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1., \
        time_step_size = 0.1, \
        g = (0., 0.), \
        output_dir="output/heat", \
        initial_values_expression = ('0.', '0.', '0.', '0.5*'+hot_wall+' - 0.5*'+cold_wall), \
        bc_expressions = [[2, ('0.'), 2, hot_wall, "topological"], [2, ('-0.'), 2, cold_wall, "topological"]])
        

def natural_convection(linearize = True, adaptive_time = True, m=20, time_step_size = 1.e-3):

    hot = 0.5
    
    cold = -0.5
    
    if linearize:
    
        hot = cold = 0.

    phase_flow.run( \
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
        
        
def run_tests():

    default()
    
    steady_lid_driven_cavity()
    
    heat()
    
    natural_convection()
    
    pass
    
    
if __name__=='__main__':

    run_tests()