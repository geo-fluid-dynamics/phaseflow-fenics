import danaila_natural_convection
#import newton_lid_driven_cavity
#import newton_steady_lid_driven_cavity

# wang2010 Figure 4.a shows that M=20 recovered the correct peak velocity, while M=40 recovers the inflection point at the center. danaila2014newton shows the result for M=80.

danaila_natural_convection.run(initial_mesh_M=10, wall_refinement_cycles=3, linearize=True, adaptive_time=True, time_step_size=1., output_dir='refine')
