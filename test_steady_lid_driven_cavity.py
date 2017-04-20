import steady_lid_driven_cavity

# For this case, the Neumann BC seems to hold for a while, and then it destabilizes and breaks.
steady_lid_driven_cavity.run(mesh_M=16, output_dir='output/steady_lid_driven_cavity_M16')
