import danaila_natural_convection

#danaila_natural_convection.run(linearize=False, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=False, time_step_size=1.e-5, final_time=1.e-4, output_dir='verify_neumann')

#For this case, the Neumann BC looks good
#danaila_natural_convection.run(linearize=True, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=False, time_step_size=1.e-5, final_time=1.e-4, output_dir='verify_neumann_linearized')

#For this case, the Neumann BC looks good
#danaila_natural_convection.run(linearize=True, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=True, time_step_size=1., final_time=1.e-4, output_dir='verify_neumann_linearized_adaptive')


# For this case, the Neumann BC looks good for a while, and then starts oscillating (and never staying close to zero gradient)
#danaila_natural_convection.run(linearize=True, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=False, time_step_size=1.e-5, final_time=1., output_dir='verify_neumann_linearized')

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
#danaila_natural_convection.run(linearize=False, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=False, time_step_size=1.e-5, final_time=1., output_dir='verify_neumann')

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
danaila_natural_convection.run(linearize=False, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=False, time_step_size=1.e-5, final_time=1., output_dir='verify_neumann')

