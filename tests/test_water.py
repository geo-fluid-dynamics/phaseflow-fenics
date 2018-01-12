from builtins import str
from .context import phaseflow

import fenics


def verify_against_michalek2003(w):
    """Verify directly against steady-state data from michalek2003 published in 

        @article{michalek2003simulations,
          title={Simulations of the water freezing process--numerical benchmarks},
          author={MICHA{\L}EK, TOMASZ and KOWALEWSKI, TOMASZ A},
          journal={Task Quarterly},
          volume={7},
          number={3},
          pages={389--408},
          year={2003}
        }
    """
    verified_solution = {'y': 0.5,
        'x': [0.00, 0.05, 0.12, 0.23, 0.40, 0.59, 0.80, 0.88, 1.00],
        'theta': [1.00, 0.66, 0.56, 0.58, 0.59, 0.62, 0.20, 0.22, 0.00]}
    
    bbt = w.function_space().mesh().bounding_box_tree()
    
    for i, true_theta in enumerate(verified_solution['theta']):
    
        p = fenics.Point(fenics.Point(verified_solution['x'][i], verified_solution['y']))
        
        if bbt.collides_entity(p):
            
            wval = w(p)
            
            theta = wval[3]
            
            assert(abs(theta - true_theta) < 2.e-2)

            
def heat_driven_cavity_water(
        initial_values = None,
        output_dir = "output/test_heat_driven_cavity_water/",
        time = 0., end_time = 10., time_step_size = 0.001,
        steady_relative_tolerance = 1.e-5):
    
    m = 40

    Ra = 2.518084e6
    
    Pr = 6.99
    
    T_h = 10. # [deg C]
    
    T_c = 0. # [deg C]
    
    theta_hot = 1.
    
    theta_cold = 0.
    
    T_f = T_fusion = 0. # [deg C]
    
    T_ref = T_fusion
    
    theta_f = theta_fusion = T_fusion - T_ref
    
    T_m = 4.0293 # [deg C]
        
    rho_m = 999.972 # [kg/m^3]
    
    w = 9.2793e-6 # [(deg C)^(-q)]
    
    q = 1.894816
    
    def rho(theta):            
        
        return rho_m*(1. - w*abs((T_h - T_c)*theta + T_ref - T_m)**q)
        
        
    def ddtheta_rho(theta):
        
        return -q*rho_m*w*abs(T_m - T_ref + theta*(T_c - T_h))**(q - 1.)* \
            fenics.sign(T_m - T_ref + theta*(T_c - T_h))*(T_c - T_h)

            
    beta = 6.91e-5 # [K^-1]
    
    def m_B(T, Ra, Pr, Re):
        
            return Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(rho(theta_f) - rho(T))/rho(theta_f)
            
            
    def ddT_m_B(T, Ra, Pr, Re):
    
        return -Ra/(Pr*Re*Re)/(beta*(T_h - T_c))*(ddtheta_rho(T))/rho(theta_f)
        

    if initial_values is None:
        
        mesh = fenics.UnitSquareMesh(fenics.mpi_comm_world(), m, m, "crossed")
        
        mixed_element = phaseflow.make_mixed_fe(mesh.ufl_cell())
        
        function_space = fenics.FunctionSpace(mesh, mixed_element)
        
        initial_values = fenics.interpolate(
            fenics.Expression(
                ("0.", "0.", "0.", "theta_hot + x[0]*(theta_cold - theta_hot)"),
                theta_hot = theta_hot, theta_cold = theta_cold,
                element = mixed_element),
            function_space)
            
    else:
    
        function_space = initial_values.function_space()
        
    solution = fenics.Function(function_space)
    
    phaseflow.run(solution = solution,
        initial_values = initial_values,
        boundary_conditions = [
            fenics.DirichletBC(function_space.sub(0), (0., 0.),
                "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)"),
            fenics.DirichletBC(function_space.sub(2), theta_hot, "near(x[0],  0.)"),
            fenics.DirichletBC(function_space.sub(2), theta_cold, "near(x[0],  1.)")],
        time = time,
        output_dir = output_dir,
        rayleigh_number = Ra,
        prandtl_number = Pr,
        stefan_number = 1.e16,
        nlp_relative_tolerance = 1.e-8,
        adaptive = False,
        m_B = m_B,
        ddT_m_B = ddT_m_B,
        stop_when_steady = True,
        steady_relative_tolerance = steady_relative_tolerance,
        time_step_size = time_step_size,
        end_time = end_time)
    
    return solution, time

    
def test_heat_driven_cavity_water():
    
    solution, time = heat_driven_cavity_water(output_dir = "output/test_heat_driven_cavity_water/",
        time_step_size = 0.001,
        end_time = 0.001)
   
    solution, time = phaseflow.read_checkpoint("output/test_heat_driven_cavity_water/checkpoint_t0.001.h5")
    
    solution, time = heat_driven_cavity_water(initial_values = solution,
        time = time,
        time_step_size = 0.002,
        end_time = 0.003,
        output_dir="output/test_heat_driven_cavity_water/restart_t0.001/")

    solution, time = phaseflow.read_checkpoint(
        "output/test_heat_driven_cavity_water/restart_t0.001/checkpoint_t0.003.h5")
    
    solution, time = heat_driven_cavity_water(initial_values = solution,
        time = time,
        time_step_size = 0.004,
        steady_relative_tolerance = 1.e-3,
        output_dir="output/test_heat_driven_cavity_water/restart_t0.003/")
    
    verify_against_michalek2003(solution)
    
    
if __name__=='__main__':

    test_heat_driven_cavity_water()
    