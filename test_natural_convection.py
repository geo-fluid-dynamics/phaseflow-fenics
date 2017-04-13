import natural_convection

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
#natural_convection.run(linearize=True, initial_mesh_M=20, adaptive_time=True, time_step_size=1.e-3, final_time=1., stop_when_steady=True, output_dir='output/linearized_natural_convection_adaptive_time')

#natural_convection.run(linearize=True, initial_mesh_M=80, adaptive_time=True, time_step_size=1.e-3, final_time=1., stop_when_steady=True, output_dir='output/linearized_natural_convection_adaptive_M80')

natural_convection.run(linearize=True, initial_mesh_M=40, adaptive_time=True, time_step_size=1.e-3, final_time=1., stop_when_steady=True, output_dir='output/linearized_natural_convection_adaptive_M40')

#natural_convection.run(linearize=False, initial_mesh_M=40, adaptive_time=False, time_step_size=1.e-3, final_time=1., stop_when_steady=True, output_dir='output/nonlinear_natural_convection_M40')