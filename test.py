import danaila_natural_convection

#danaila_natural_convection.run(linearize=True, initial_mesh_M=20, wall_refinement_cycles=5, adaptive_time=True, time_step_size=1., output_dir='refine')

danaila_natural_convection.run(linearize=False, initial_mesh_M=1, wall_refinement_cycles=8, adaptive_time=False, time_step_size=1.e-5, final_time=1.e-4, output_dir='verify_neumann')
