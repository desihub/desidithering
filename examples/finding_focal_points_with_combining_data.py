import astropy.units as u
import numpy as np
import yaml
import random
import math
import sys
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
from astropy import table
# add dithering module to path and load it
sys.path.append('../py')
import dithering as Dithering
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
    g = amplitude * np.exp( - ( (x-xo)**2/(2*sigma_x**2) + (y-yo)**2/(2*sigma_y**2) ) )
    return g.ravel()

def twoD_ellipticGaussian(xy_tuple, amplitude, xo, yo, sigma_1, sigma_2, phi):
    (x, y) = xy_tuple
    amplitude = float(amplitude)
    xo = float(xo)
    yo = float(yo)
    sigma_1 = float(sigma_1)
    sigma_2 = float(sigma_2)
    phi     = float(phi)
    A = (np.cos(phi) / sigma_1)**2 + (np.sin(phi) / sigma_2)**2
    B = (np.sin(phi) / sigma_1)**2 + (np.cos(phi) / sigma_2)**2
    C = 2 * np.sin(phi) * np.cos(phi)*(1/sigma_1**2 - 1/sigma_2**2)
    g = amplitude * np.exp( -0.5 * ( A*(x-xo)**2 + B*(y-yo)**2 + C*(x-xo)*(y-yo) ) )
    return g.ravel()

def calc_signal_at_position():
    return None

def cost_function():
    return None

def fit_simulation(x, y, signal, SNR):
    coordinates = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
    data = np.array(SNR).ravel()
    start_index = np.argmax(data)
    initial_guess = (max(SNR), x[start_index], y[start_index], 50., 50., math.pi/100.)

    try:
        popt, pcov = opt.curve_fit(twoD_ellipticGaussian,
                                   coordinates,
                                   data,
                                   p0=initial_guess,
                                   bounds=([0.1, -100, -100, 1, 1, -math.pi], [40, 100, 100, 100, 100, math.pi]) )
    except:
        return -9999, -9999, -9999, -9999, -9999
    return popt[1], popt[2], popt[3], popt[4], popt[5]

parser = argparse.ArgumentParser(description="Script to find the optimal focal point given a set of parameters using the previously generated simulations")
parser.add_argument("--step-size",      dest="stepsize",         type=float, required=True, help="Step sizes for the optimization algorithm", nargs="+")
parser.add_argument("--offset-rms",     dest="randomoffset",     type=float, required=True, help="Random offsets to be introducted")
parser.add_argument("--seeing_offsets_rms",  dest="seeing_offsets",    type=float,  default=0.0, help="RMS of the seeing offsets")
parser.add_argument("--dithering_pattern",   dest="pattern",           type=int,    default=0,   help="Dithering pattern, 0: triangular, 1: rectangular")
parser.add_argument("--output",              dest="outfname",          type=str,    default="results")
parsed_args = parser.parse_args()

search_radia      = parsed_args.stepsize
random_           = parsed_args.randomoffset
seeing_offset_rms = parsed_args.seeing_offsets
outfname          = parsed_args.outfname
pattern           = parsed_args.pattern

main_directory_path = "../data/{}um_RMS".format(random_)
files = []
curr_pattern = ["triangle", "rectangle"][pattern]
for search_radius in search_radia:
    curr_fname = "{}/{}um/result_{}_seeing{}_blur_wlenoffset-randoffset.fits".format(main_directory_path, search_radius, curr_pattern, seeing_offset_rms) 
    files.append(curr_fname)
num_files = len(files)

# generate the fits file primary header
prihdr = fits.Header()
prihdr['COMMENT'] = "The analysis was done combining the following files:"
for filename in files:
    prihdr["COMMENT"] = "{}".format(filename)
prihdu  = fits.PrimaryHDU(header=prihdr)
hdulist = fits.HDUList([prihdu])
prev_num_sources = None

data= fits.open(files[0])[1].data
num_sources = len(data['known_offset_x'])
# here is what we will save in the fits file
bore_alt = np.zeros(num_sources)
bore_az  = np.zeros(num_sources)
src_alt = np.zeros(num_sources)
src_az  = np.zeros(num_sources)
known_offset_x = np.zeros(num_sources)
known_offset_y = np.zeros(num_sources)
known_systematic_x = np.zeros(num_sources)
known_systematic_y = np.zeros(num_sources)
magnitudes = np.zeros((num_sources, 5))
calc_offset_x = np.zeros(num_sources)
calc_offset_y = np.zeros(num_sources)
calc_sigma_x = np.zeros(num_sources)
calc_sigma_y = np.zeros(num_sources)
known_snr_r = np.zeros(num_sources)
known_snr_b = np.zeros(num_sources)
known_snr_z = np.zeros(num_sources)
calc_snr_r = np.zeros(num_sources)
calc_snr_b = np.zeros(num_sources)
calc_snr_z = np.zeros(num_sources)
dither_xs  = np.zeros((num_sources,9*num_files))
dither_ys  = np.zeros((num_sources,9*num_files))
calc_snrs    = np.zeros((num_sources,9*num_files))
calc_signals = np.zeros((num_sources,9*num_files))
calc_signal_r = np.zeros(num_sources)
calc_signal_b = np.zeros(num_sources)
calc_signal_z = np.zeros(num_sources)
known_signal_r = np.zeros(num_sources)
known_signal_b = np.zeros(num_sources)
known_signal_z = np.zeros(num_sources)
focal_xs   = np.zeros(num_sources)
focal_ys   = np.zeros(num_sources)

for iFileID in range(num_files):
    filename = files[iFileID]
    # read the datafile and do a sanity check
    data = fits.open(filename)[1].data
    num_sources = len(data['known_offset_x'])
    if prev_num_sources is None:
        prev_num_sources = num_sources
    else:
        if prev_num_sources != num_sources:
            print("mismatch in number of sources in files")
            break
        else:
            prev_num_sources = num_sources

    # define the source to be observed
    for j in range(num_sources):
        xs, ys, signals, SNRs = data['dither_pos_x'][j], data['dither_pos_y'][j], data['calc_signals'][j], data['calc_snrs'][j]
        dither_xs[j, iFileID*9:iFileID*9+9] = xs
        dither_ys[j, iFileID*9:iFileID*9+9] = ys
        calc_signals[j, iFileID*9:iFileID*9+9] = signals
        calc_snrs[j, iFileID*9:iFileID*9+9]    = SNRs
        
for iSource in range(num_sources):
    data = fits.open(files[0])[1].data
    xs = dither_xs[iSource]
    ys = dither_ys[iSource]
    signals = calc_signals[iSource]
    SNRs = calc_snrs[iSource]
    opt_x_offset, opt_y_offset, opt_x_sigma, opt_y_sigma, opt_phi= fit_simulation(xs, ys, signals, SNRs)
    
    focal_xs[iSource] = data['focal_x'][iSource]
    focal_ys[iSource] = data['focal_y'][iSource]
    known_offset_x[iSource] = data['known_offset_x'][iSource]
    known_offset_y[iSource] = data['known_offset_y'][iSource]
    magnitudes[iSource] = data['mag'][iSource]
    
    calc_offset_x[iSource] = opt_x_offset
    calc_offset_y[iSource] = opt_y_offset
    calc_sigma_x[iSource] = opt_x_sigma
    calc_sigma_y[iSource] = opt_y_sigma

    del xs, ys, SNRs, signals
    del opt_x_offset, opt_y_offset, opt_x_sigma, opt_y_sigma
                    
hodor = fits.Header()
hodor['seeing_offset_rms']  = seeing_offset_rms
hodor['search_radiuss']     = search_radius
hodor['random_offset_rms']  = random_
    
thdu = fits.BinTableHDU.from_columns(
    [fits.Column(name="known_offset_x",     array=known_offset_x, format="E"),
     fits.Column(name="known_offset_y",     array=known_offset_y, format="E"),
     fits.Column(name="calc_offset_x",      array=calc_offset_x, format="E"),
     fits.Column(name="calc_offset_y",      array=calc_offset_y, format="E"),
     fits.Column(name="calc_sigma_x",       array=calc_sigma_x, format="E"),
     fits.Column(name="calc_sigma_y",       array=calc_sigma_y, format="E"),
     fits.Column(name="calc_snrs",          array=calc_snrs,  format="{}E".format(9*num_files)),
     fits.Column(name="dither_pos_x",       array=dither_xs,  format="{}E".format(9*num_files)),
     fits.Column(name="dither_pos_y",       array=dither_ys,  format="{}E".format(9*num_files)),
     fits.Column(name="calc_signals",       array=calc_signals,  format="{}E".format(9*num_files)),
     fits.Column(name="focal_x",            array=focal_xs, format='E'),
     fits.Column(name="focal_y",            array=focal_ys, format="E"),
     fits.Column(name="mag",                array=magnitudes, format="5E"),
    ], header=hodor)
hdulist.append(thdu)
    
try:
    os.mkdir("../data/{}um_RMS".format(random_))
except:
    print("main folder exists... checking the subfolder...")

try:
    os.mkdir("../data/{}um_RMS/{}um".format(random_, search_radius))
except:
    print("subfolder exists.. moving on to saving the file")

temp_filename = "../data/{}um_RMS/{}_combined.fits".format(random_, outfname)
filename = temp_filename
trial = 1
while True:
    if os.path.exists(filename):
        filename = temp_filename.split(".fits")[0]+"_{}".format(trial)+".fits"
        trial = trial + 1
    else:
        break
hdulist.writeto(filename)
