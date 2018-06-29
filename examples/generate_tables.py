import numpy as np
import matplotlib.pyplot as plt
import random
import sys

offset_rms = float(sys.argv[1])
num_fibers = 5000
focal_plane_radius = 410

x_offsets = np.random.normal(0.0, scale=offset_rms, size=num_fibers)
y_offsets = np.random.normal(0.0, scale=offset_rms, size=num_fibers)

r_pos   = np.random.uniform(0, focal_plane_radius, size=num_fibers)
phi_pos = np.random.uniform(-np.pi, np.pi, size=num_fibers)
x_pos   = r_pos * np.cos(phi_pos)
y_pos   = r_pos * np.sin(phi_pos)

np.savez('tabled_values_{}.npz'.format(offset_rms), x_pos=x_pos, y_pos=y_pos, x_offsets=x_offsets, y_offsets=y_offsets)

