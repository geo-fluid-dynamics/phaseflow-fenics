import natural_convection

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
natural_convection.run(linearize=True, initial_mesh_M=20, adaptive_time=True, time_step_size=1., final_time=1., output_dir='output/natural_convection')

