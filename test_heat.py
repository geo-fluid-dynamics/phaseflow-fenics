import natural_convection


natural_convection.run(g=(0., 0.), linearize=True, initial_mesh_M=4, adaptive_time=True, time_step_size=1.e-3, final_time=1., stop_when_steady=True, output_dir='output/linearized_heat_adaptive_M4')
