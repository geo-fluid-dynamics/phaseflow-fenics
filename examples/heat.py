import phaseflow

from fenics import UnitSquareMesh

    
def run(m=16):

    hot_wall = 'near(x[0], 0.)'
    
    cold_wall = 'near(x[0], 1.)'

    adiabatic_walls = 'near(x[1],  0.) | near(x[1], 1.)'

    phaseflow.run(linearize = True, \
        adaptive_time = True, \
        mesh = UnitSquareMesh(m, m, "crossed"), \
        final_time = 1., \
        time_step_size = 0.1, \
        g = (0., 0.), \
        output_dir="output/heat", \
        initial_values_expression = ('0.', '0.', '0.', '0.5*'+hot_wall+' - 0.5*'+cold_wall), \
        bc_expressions = [[2, ('0.'), 2, hot_wall, "topological"], [2, ('-0.'), 2, cold_wall, "topological"]])

        
if __name__=='__main__':

    run()