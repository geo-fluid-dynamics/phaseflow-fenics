import danaila_natural_convection

mesh_M = 160
time_step_size = 1.e-3

danaila_natural_convection.run(mesh_M=mesh_M, time_step_size=time_step_size, final_time=time_step_size)