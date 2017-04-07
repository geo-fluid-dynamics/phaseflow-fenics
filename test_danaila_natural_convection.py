import danaila_natural_convection

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
danaila_natural_convection.run(mesh_M=20, time_step_size=1.e-3, final_time=1., output_dir='output/danaila_natural_convection')

