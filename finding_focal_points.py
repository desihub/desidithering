import astropy.units as u
import numpy as np
import yaml
import random
import sys
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
# add dithering module to path and load it
sys.path.append('./py')
import dithering
import extract_spectrum as es

# general plotting options 
rcParams.update({'figure.autolayout': True})
my_cmap = plt.cm.jet
my_cmap.set_under('white',.01)
my_cmap.set_over('white', 300)
my_cmap.set_bad('white')

def twoD_Gaussian(xy_tuple, amplitude, xo, yo):
    (x, y) = xy_tuple
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    sigma_x = random_
    sigma_y = random_
    g = amplitude * np.exp( - ( (x-xo)**2/(2*sigma_x**2) + (y-yo)**2/(2*sigma_y**2) ) )
    return g.ravel()
    
def run_simulation(dithering, source, source_alt, source_az, boresight_alt, boresight_az):

    # generate the random offset but not reveal it to user until the end
    x_offset = random.gauss(0, random_)# * random_
    y_offset = random.gauss(0, random_)# * random_
    
    dithering.set_boresight_position(boresight_alt*u.deg, boresight_az*u.deg)
    dithering.set_source_position(source_alt*u.deg, source_az*u.deg)
    dithering.set_focal_plane_position()
    dithering.set_theta_0(-5.*u.deg)
    dithering.set_phi_0(10.*u.deg)
    SNR = []
    x   = []
    y   = []

    # initial point
    dithering.place_fiber([x_offset, y_offset])
    dithering.run_simulation(source_type, *source, report=False)
    SNR.append(np.median(dithering.SNR['b'][0]))
    x.append(0)
    y.append(0)
    #print("fiber placement before test: {0}".format(dithering.fiber_placement))
    
    # dithering
    search_radia = [random_*1., random_*2.]
    num_dithering = 3
    for i in range(num_dithering):
        x_dither = search_radia[0] * np.cos((360./num_dithering)*i*u.deg)
        y_dither = search_radia[0] * np.sin((360./num_dithering)*i*u.deg)
        dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
        dithering.run_simulation(source_type, *source, report=False)
        SNR.append(np.median(dithering.SNR['b'][0]))
        x.append(x_dither)
        y.append(y_dither)
        x_dither = search_radia[1] * np.cos((360./num_dithering)*i*u.deg)
        y_dither = search_radia[1] * np.sin((360./num_dithering)*i*u.deg)
        dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
        dithering.run_simulation(source_type, *source, report=False)
        SNR.append(np.median(dithering.SNR['b'][0]))
        x.append(x_dither)
        y.append(y_dither)

    #plt.hist2d(x, y, weights=SNR, bins=num_dithering*2, cmap=my_cmap, vmin=0.05)
    #plt.xlabel('x position [um]')
    #plt.ylabel('y position [um]')
    #plt.colorbar(label='SNR')
    #plt.show()

    coordinates = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
    data = np.array(SNR).ravel()
    initial_guess = (20., 0., 0.)
    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, coordinates, data, p0=initial_guess)
    except:
        #print("NO CONVERGENCE")
        return -9999, -9999, -9999, -9999
    #print("Optimization found the following:")
    dithering.place_fiber([x_offset+popt[1], y_offset+popt[2]])
    dithering.run_simulation(source_type, *source, report=False)
    #print("If there was not error (offset):")
    #dithering.place_fiber([0., 0.])
    #dithering.run_simulation(source_type, *source, report=True)
    #print("====================================================")
    return x_offset, y_offset, popt[1], popt[2]
    
config_file = "./config/desi-noblur-nooffset.yaml"
dithering = dithering.dithering(config_file=config_file)
random_       = 50.
num_sources   = 200
num_pointings = 2
num_total     = num_pointings * num_sources - 1
results_b     = []
real_values_b = []
results_r     = []
real_values_r = []
results_z     = []
real_values_z = []

try:
    import progressbar
    bar = progressbar.ProgressBar(max_value=num_total,
                                  widgets=[ ' [', progressbar.Timer(), '] ',
                                            progressbar.Bar(),
                                            ' (', progressbar.ETA(), ') ',])
    bar.update(0)
except:
    bar = None

# generate the fits file primary header
prihdr = fits.Header()
prihdr['COMMENT'] = "The config file used is {}.".format(config_file)
prihdr["COMMENT"] = "From a uniform distribution with [0, 30), random pointings were chosen"
prihdr['COMMENT'] = "For each boresight altitude and azimuth, there are {} sources randomly located (-1, 1) degrees of the boresight.".format(num_sources)
prihdr["COMMENT"] = "For random x and y offsets, {} um was assumed.".format(random_)
prihdu  = fits.PrimaryHDU(header=prihdr)
hdulist = fits.HDUList([prihdu])

for i in range(num_pointings):
    # define the observing conditions
    boresight_alt = random.uniform(0., 30.)
    boresight_az  = random.uniform(0., 30.)

    # here is what we will save in the fits file
    bore_alt = np.zeros(num_sources)
    bore_az  = np.zeros(num_sources)
    src_alt = np.zeros(num_sources)
    src_az  = np.zeros(num_sources)
    known_offset_x = np.zeros(num_sources)
    known_offset_y = np.zeros(num_sources)
    calc_offset_x = np.zeros(num_sources)
    calc_offset_y = np.zeros(num_sources)
    known_snr_r = np.zeros(num_sources)
    known_snr_b = np.zeros(num_sources)
    known_snr_z = np.zeros(num_sources)
    calc_snr_r = np.zeros(num_sources)
    calc_snr_b = np.zeros(num_sources)
    calc_snr_z = np.zeros(num_sources)
    
    # define the source to be observed
    source_type       = "qso"
    half_light_radius = 0.3
    for j in range(num_sources):

        if bar is not None:
            bar.update((i+1)*j)

        source_alt = boresight_alt + random.uniform(-1., 1.)
        source_az  = boresight_az  + random.uniform(-1., 1.)

        bore_alt[j] = boresight_alt
        bore_az[j]  = boresight_az
        src_alt[j] = source_alt
        src_az[j]  = source_az
        
        # Generate the source to be observed
        if source_type == "qso" or source_type == "QSO":
            source    = dithering.generate_source(disk_fraction=0., bulge_fraction=0., 
                                                  half_light_disk=0., half_light_bulge=0.)
        elif source_type == "elg" or source_type == "ELG":
            source    = dithering.generate_source(disk_fraction=1., bulge_fraction=0., 
                                                  half_light_disk=half_light_radius, half_light_bulge=0.)
        elif source_type == "lrg" or source_type == "LRG":
            source    = dithering.generate_source(disk_fraction=0., bulge_fraction=1., 
                                                  half_light_disk=0., half_light_bulge=half_light_radius)

        w_in, f_in = es.get_random_spectrum("STD_FSTAR")
        dithering.desi.source.update_in("F type star", "star", w_in*u.angstrom, f_in*1e-17*u.erg/(u.angstrom*u.cm*u.cm*u.s))
        dithering.desi.source.update_out()
    
        x_offset, y_offset, opt_x_offset, opt_y_offset  = run_simulation(dithering, source, source_alt, source_az, boresight_alt, boresight_az)
        known_offset_x[j] = x_offset
        known_offset_y[j] = y_offset
        calc_offset_x[j] = opt_x_offset
        calc_offset_y[j] = opt_y_offset
        
        calc_snr_b[j] = (np.median(dithering.SNR['b'][0]))
        calc_snr_r[j] = (np.median(dithering.SNR['r'][0]))
        calc_snr_z[j] = (np.median(dithering.SNR['z'][0]))
        dithering.place_fiber([0., 0.])
        dithering.run_simulation(source_type, *source, report=False)
        known_snr_b[j] = (np.median(dithering.SNR['b'][0]))
        known_snr_r[j] = (np.median(dithering.SNR['r'][0]))
        known_snr_z[j] = (np.median(dithering.SNR['z'][0]))
                
    thdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="boresight_alt", array=bore_alt, format="E"),
         fits.Column(name="boresight_az", array=bore_az, format="E"),
         fits.Column(name="source_alt", array=src_alt, format="E"),
         fits.Column(name="source_az", array=src_az, format="E"),
         fits.Column(name="known_offset_x", array=known_offset_x, format="E"),
         fits.Column(name="known_offset_y", array=known_offset_y, format="E"),
         fits.Column(name="calc_offset_x", array=calc_offset_x, format="E"),
         fits.Column(name="calc_offset_y", array=calc_offset_y, format="E"),
         fits.Column(name="known_snr_b", array=known_snr_b, format="E"),
         fits.Column(name="known_snr_r", array=known_snr_r, format="E"),
         fits.Column(name="known_snr_z", array=known_snr_z, format="E"),
         fits.Column(name="calc_snr_b", array=calc_snr_b, format="E"),
         fits.Column(name="calc_snr_r", array=calc_snr_r, format="E"),
         fits.Column(name="calc_snr_z", array=calc_snr_z, format="E"),
        ])
    hdulist.append(thdu)

temp_filename = "results.fits"
filename = temp_filename
trial = 1
while True:
    if os.path.exists(filename):
        filename = temp_filename.split(".")[0]+"_{}".format(trial)+".fits"
        trial = trial + 1
    else:
        break
hdulist.writeto(filename)
