import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import math
import astropy.units as u
sys.path.append('../py')
import extract_spectrum as es

offset_rms = float(sys.argv[1])

# some constants
num_fibers = 2000
focal_plane_radius = 410
num_theta = 100

# random un-correlated offsets for each fiber
x_offsets = np.random.normal(0.0, scale=offset_rms, size=num_fibers)
y_offsets = np.random.normal(0.0, scale=offset_rms, size=num_fibers)

# calculation for the fiber positions
# they are linearly distributed but basically can follow any pattern
#r_pos   = np.linspace(0, focal_plane_radius, int(num_fibers/num_theta))
#r_pos   = np.random.triangular(0, focal_plane_radius, focal_plane_radius, int(num_fibers/num_theta))
r_pos   = focal_plane_radius*np.random.power(2, size=num_fibers)
phi_pos = 2*math.pi*np.random.uniform(size=num_fibers)#np.linspace(-np.pi, np.pi, num_theta)
x_pos   = r_pos * np.cos(phi_pos)
y_pos   = r_pos * np.sin(phi_pos)
x_pos   = x_pos.reshape(1, -1)[0]
y_pos   = y_pos.reshape(1, -1)[0]

# get a random spectrum from the catalog
# this is now done at this level for different reasons:
# major reason is that we want to be able to analyze the same
# stars at the same fiber locations with different configurations
wlens  = []
fluxes = []
mags   = []
for i in range(num_fibers):
    if i%5==0:
        w_in, f_in, mag = es.get_random_spectrum("STD_FSTAR")
        if mag[0]>21. or mag[0]<9.:
            i -= 1
            continue
    wlens.append(w_in)
    fluxes.append(f_in)
    mags.append(mag)
    if i%20==0:
        print("{}/{}".format(i, num_fibers))

np.savez('tabled_values_{}_2.npz'.format(offset_rms),
         x_pos=x_pos, y_pos=y_pos,
         x_offsets=x_offsets, y_offsets=y_offsets,
         wlens=wlens, fluxes=fluxes, mags=mags)
