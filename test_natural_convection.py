import natural_convection

natural_convection.run(linearize=True, mesh = UnitSquareMesh(40, 40, "crossed"), adaptive_time=True, time_step_size=1.e-3, final_time=1., stop_when_steady=True, output_dir='output/linearized_natural_convection_adaptive_M40')
