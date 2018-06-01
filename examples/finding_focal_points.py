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
from astropy import table
# add dithering module to path and load it
sys.path.append('../py')
import dithering
import extract_spectrum as es
import argparse
import time

is_plot = False

# general plotting options 
rcParams.update({'figure.autolayout': True})
my_cmap = plt.cm.jet
my_cmap.set_under('white',.01)
my_cmap.set_over('white', 300)
my_cmap.set_bad('white')

def twoD_Gaussian(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y):
    (x, y) = xy_tuple
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    #sigma_x = random_
    #sigma_y = random_
    g = amplitude * np.exp( - ( (x-xo)**2/(2*sigma_x**2) + (y-yo)**2/(2*sigma_y**2) ) )
    return g.ravel()
    
def run_simulation(dithering, source, source_alt, source_az, boresight_alt, boresight_az, random_seeing_offsets, random_airmass_offsets):
    check_angles_primary   = { 0: [-30., -150.],
                               1: [-30., 90],
                               2: [-150., 90.],
                               3: [330., 210.] }
    """
    check_angles_secondary = { 0: {4: [30, -90],  5: [150, -90]},
                               1: {4: [30, -90],  5: [30, 150]},
                               2: {4: [150, -90], 5: [30, 150]},
                               3: {4: [30, -90],  5: [150, -90]} }
    """
    check_angles_secondary = { 0: {4: [60.0,  240.0, 330.0], 5: [120.0, 210.0, 300.0]},
                               1: {4: [60.0,  240.0, 330.0], 5: [0.0,   90.0,  180.0]},
                               2: {4: [120.0, 210.0, 300.0], 5: [0.0,   90.0,  180.0]},
                               3: {4: [60.0,  240.0, 330.0], 5: [120.0, 210.0, 300.0]} }
    # generate the random offset but not reveal it to user until the end
    x_offset = random.gauss(0, random_)
    y_offset = random.gauss(0, random_)
    
    dithering.set_boresight_position(boresight_alt*u.deg, boresight_az*u.deg)
    dithering.set_source_position(source_alt*u.deg, source_az*u.deg)
    dithering.set_focal_plane_position()
    dithering.set_theta_0(-5.*u.deg)
    dithering.set_phi_0(10.*u.deg)
    SNR = []
    x   = []
    y   = []

    focal_x = dithering.focal_x[0]
    focal_y = dithering.focal_y[0]
    
    # initial point
    dithering.place_fiber([x_offset, y_offset])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[0], airmass_offset=random_airmass_offsets[0])
    SNR.append(np.median(dithering.SNR['b'][0]))
    x.append(0)
    y.append(0)
    #print("fiber placement before test: {0}".format(dithering.fiber_placement))

    # dithering starts here
    # it is a very simple method to find the maximum point
    # the optimal is with 6 measurements (including the initial measurement)
    # -- first dither
    x_dither = search_radius*np.cos(30.*u.deg)
    y_dither = search_radius*np.sin(30.*u.deg)
    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[1], airmass_offset=random_airmass_offsets[1])
    SNR.append(np.median(dithering.SNR['b'][0]))
    x.append(x_dither)
    y.append(y_dither)
    # -- second dither
    x_dither = -1*search_radius*np.cos(30.*u.deg)
    y_dither = search_radius*np.sin(30.*u.deg)
    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[2], airmass_offset=random_airmass_offsets[2])
    SNR.append(np.median(dithering.SNR['b'][0]))
    x.append(x_dither)
    y.append(y_dither)
    # -- third dither
    x_dither = 0. 
    y_dither = -1*search_radius
    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[3], airmass_offset=random_airmass_offsets[3])
    SNR.append(np.median(dithering.SNR['b'][0]))
    x.append(x_dither)
    y.append(y_dither)
    # -- next two dithering depends on the maximum among the ones searches
    max_idx = np.argmax(SNR)
    new_angles_primary   = check_angles_primary[max_idx]
    for i in range(2):
        x_dither = search_radius*np.cos(new_angles_primary[i]*u.deg)+x[max_idx]
        y_dither = search_radius*np.sin(new_angles_primary[i]*u.deg)+y[max_idx]
        dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
        dithering.run_simulation(source_type, *source, report=False, \
                                 seeing_fwhm_ref_offset=random_seeing_offsets[4+i], airmass_offset=random_airmass_offsets[4+i])
        SNR.append(np.median(dithering.SNR['b'][0]))
        x.append(x_dither)
        y.append(y_dither)
    if(SNR[4]>SNR[5]):
        new_angles_secondary = check_angles_secondary[max_idx][4]
        max_idx2 = 4
    else:
        new_angles_secondary = check_angles_secondary[max_idx][5]
        max_idx2 = 5
    for i in range(3):
        x_dither = .75*search_radius*np.cos(new_angles_secondary[i]*u.deg) + x[max_idx2]
        y_dither = .75*search_radius*np.sin(new_angles_secondary[i]*u.deg) + y[max_idx2]
        dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
        dithering.run_simulation(source_type, *source, report=False, \
                                 seeing_fwhm_ref_offset=random_seeing_offsets[6+i], airmass_offset=random_airmass_offsets[6+i])
        SNR.append(np.median(dithering.SNR['b'][0]))
        x.append(x_dither)
        y.append(y_dither)

    if is_plot:
        plt.plot(x[0], y[0], 'o')
        plt.plot(x[1], y[1], '*')
        plt.plot(x[2], y[2], '*')
        plt.plot(x[3], y[3], '*')
        plt.plot(x[4], y[4], 'v')
        plt.plot(x[5], y[5], 'v')
        plt.plot(x[6], y[6], 's')
        plt.plot(x[7], y[7], 's')
        plt.plot(x[8], y[8], 's')
        plt.plot(-x_offset, -y_offset, 'p')
        plt.show()
    
    coordinates = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
    data = np.array(SNR).ravel()
    initial_guess = (20., 0., 0., random_, random_)
    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, coordinates, data, p0=initial_guess)
    except:
        #print("NO CONVERGENCE")
        return -9999, -9999, -9999, -9999, -9999, focal_x, focal_y
    #print("Optimization found the following:")
    ####dithering.place_fiber([x_offset+popt[1], y_offset+popt[2]])
    ####dithering.run_simulation(source_type, *source, report=False)
    #print("If there was not error (offset):")
    #dithering.place_fiber([0., 0.])
    #dithering.run_simulation(source_type, *source, report=True)
    #print("====================================================")
    #print(x_offset, y_offset, popt[1], popt[2])
    return x_offset, y_offset, popt[1], popt[2], SNR, focal_x, focal_y

parser = argparse.ArgumentParser(description="Script to find the optimal focal point given a set of parameters")
parser.add_argument("-c",               dest="config", default="../config/desi-noblur-nooffset.yaml")
parser.add_argument("--step-size",      dest="stepsize",         type=float, required=True)
parser.add_argument("--offset-rms",     dest="randomoffset",     type=float, required=True)
parser.add_argument("--systematic",     dest="systematicoffset", type=float, default=[0.0, 0.0], nargs=2)
parser.add_argument("--half-light-radius",   dest="half_light_radius", type=float, default=0.5)
parser.add_argument("--seeing_offsets_rms",  dest="seeing_offsets",    type=float, default=0.0, help="RMS of the seeing offsets")
parser.add_argument("--airmass_offsets_rms", dest="airmass_offsets",   type=float, default=0.0, help="RMS of the airmass offsets")
parser.add_argument("--number_of_fibers",    dest="num_sources",       type=int,   default=200)
parsed_args = parser.parse_args()

config_file       = parsed_args.config
dithering         = dithering.dithering(config_file=config_file)
search_radius     = parsed_args.stepsize
random_           = parsed_args.randomoffset
systematic_       = parsed_args.systematicoffset
half_light_radius = parsed_args.half_light_radius
num_sources       = parsed_args.num_sources
seeing_offset_rms = parsed_args.seeing_offsets
airmass_offset_rms= parsed_args.airmass_offsets

num_pointings = 1
num_total     = num_pointings * num_sources - 1
results_b     = []
real_values_b = []
results_r     = []
real_values_r = []
results_z     = []
real_values_z = []

if ( os.path.isdir("../data/{}um".format(search_radius)) ):
    print("directory path exists")
else:
    os.mkdir("../data/{}um".format(search_radius))

try:
    import progressbar
    bar = progressbar.ProgressBar(max_value=num_total,
                                  widgets=[ ' [', progressbar.Timer(), '] ',
                                            progressbar.Bar(),
                                            ' (', progressbar.ETA(), ') ',])
    bar.update(0)
except:
    bar = None

# generate the random airmass offsets to be introduced
# depending on the arguments given, they will be either all zeros
# or they will have a gaussian distribution with a mean given in the config file
# and sigma provided by David's technical note
# same procedure will be followed for the seeing (using seeing_fwhm_ref)
# since all the measurements of an exposure of 500 fibers would be at the same time,
# there are only 9 offset values for each iteration of the algorithm
random_seeing_fwhm_ref_offsets = np.random.normal(0., seeing_offset_rms,  9)*u.arcsec
random_airmass_offsets         = np.random.normal(0., airmass_offset_rms, 9)

# generate the fits file primary header
prihdr = fits.Header()
prihdr['COMMENT'] = "The config file used is {}.".format(config_file)
prihdr["COMMENT"] = "From a uniform distribution with [0, 30), random pointings were chosen"
prihdr['COMMENT'] = "For each boresight altitude and azimuth, there are {} sources randomly located (-1, 1) degrees of the boresight.".format(num_sources)
prihdr["COMMENT"] = "For random x and y offsets, {} um was assumed.".format(random_)
prihdr["HLR"]     = half_light_radius
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
    known_systematic_x = np.zeros(num_sources)
    known_systematic_y = np.zeros(num_sources)
    calc_offset_x = np.zeros(num_sources)
    calc_offset_y = np.zeros(num_sources)
    known_snr_r = np.zeros(num_sources)
    known_snr_b = np.zeros(num_sources)
    known_snr_z = np.zeros(num_sources)
    calc_snr_r = np.zeros(num_sources)
    calc_snr_b = np.zeros(num_sources)
    calc_snr_z = np.zeros(num_sources)
    calc_snrs  = np.zeros((num_sources,9))
    focal_xs   = np.zeros(num_sources)
    focal_ys   = np.zeros(num_sources)
    
    # define the source to be observed
    source_type       = "qso"
    curr_time         = time.time()
    for j in range(num_sources):
        
        if bar is not None:
            bar.update((i+1)*j)
        else:
            if j%20==0:
                print("{}: {}/{}".format(-curr_time+time.time(), j, num_sources))
                curr_time = time.time()
                
        source_alt = boresight_alt + random.uniform(-.45, .45)
        source_az  = boresight_az  + random.uniform(-.45, .45)

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

        x_offset, y_offset, opt_x_offset, opt_y_offset, SNR, focal_x, focal_y  = run_simulation(dithering, \
                                                                                                source, source_alt, source_az, \
                                                                                                boresight_alt, boresight_az, \
                                                                                                random_seeing_fwhm_ref_offsets, \
                                                                                                random_airmass_offsets)
        
        known_offset_x[j] = x_offset
        known_offset_y[j] = y_offset
        known_systematic_x[j] = systematic_[0]
        known_systematic_y[j] = systematic_[1]
        calc_offset_x[j] = opt_x_offset
        calc_offset_y[j] = opt_y_offset

        focal_xs[j] = focal_x.value
        focal_ys[j] = focal_y.value
        
        calc_snr_b[j] = (np.median(dithering.SNR['b'][0]))
        calc_snr_r[j] = (np.median(dithering.SNR['r'][0]))
        calc_snr_z[j] = (np.median(dithering.SNR['z'][0]))
        calc_snrs[j]  = np.array(SNR)
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
         fits.Column(name="known_systematic_x", array=known_systematic_x, format="E"),
         fits.Column(name="known_systematic_y", array=known_systematic_y, format="E"),
         fits.Column(name="calc_offset_x", array=calc_offset_x, format="E"),
         fits.Column(name="calc_offset_y", array=calc_offset_y, format="E"),
         fits.Column(name="known_snr_b", array=known_snr_b, format="E"),
         fits.Column(name="known_snr_r", array=known_snr_r, format="E"),
         fits.Column(name="known_snr_z", array=known_snr_z, format="E"),
         fits.Column(name="calc_snr_b", array=calc_snr_b, format="E"),
         fits.Column(name="calc_snr_r", array=calc_snr_r, format="E"),
         fits.Column(name="calc_snr_z", array=calc_snr_z, format="E"),
         fits.Column(name="calc_snrs", array=calc_snrs,  format="9E"),
         fits.Column(name="focal_x", array=focal_xs, format='E'),
         fits.Column(name="focal_y", array=focal_ys, format="E"),
        ])
    hdulist.append(thdu)
    
try:
    os.mkdir("../data/{}um_RMS".format(random_))
except:
    print("main folder exists... checking the subfolder...")

try:
    os.mkdir("../data/{}um_RMS/{}um".format(random_, search_radius))
except:
    print("subfolder exists.. moving on to saving the file")
    
temp_filename = "../data/{}um_RMS/{}um/results.fits".format(random_, search_radius)
filename = temp_filename
trial = 1
while True:
    if os.path.exists(filename):
        filename = temp_filename.split(".fits")[0]+"_{}".format(trial)+".fits"
        trial = trial + 1
    else:
        break
hdulist.writeto(filename)
