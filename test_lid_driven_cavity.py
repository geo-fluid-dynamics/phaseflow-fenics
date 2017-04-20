import lid_driven_cavity

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
lid_driven_cavity.run(mesh_M=16, time_step_size=1.e-3, final_time = 2.e-3, output_dir='output/lid_driven_cavity_M16')
