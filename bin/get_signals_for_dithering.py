#!/bin/python

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

from desidithering import dithering as Dithering
from desidithering import extract_spectrum as es
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


check_angles_primary   = { 0: [-30., -150.],
                           1: [-30., 90],
                           2: [-150., 90.],
                           3: [330., 210.] }
check_angles_secondary = { 0: {4: [60.0,  240.0, 330.0], 5: [120.0, 210.0, 300.0]},
                           1: {4: [60.0,  240.0, 330.0], 5: [0.0,   90.0,  180.0]},
                           2: {4: [120.0, 210.0, 300.0], 5: [0.0,   90.0,  180.0]},
                           3: {4: [60.0,  240.0, 330.0], 5: [120.0, 210.0, 300.0]} }
def run_simulation_triangulate(dithering, source, source_id, random_seeing_offsets, random_airmass_offsets, pos_rms):
    # generate the random offset but not reveal it to user until the end
    x_offset = tabled_x_offsets[source_id]
    y_offset = tabled_y_offsets[source_id]
    if pos_rms > 0:
        pos_x_err = np.random.normal(loc=0, scale=pos_rms, size=9)
        pos_y_err = np.random.normal(loc=0, scale=pos_rms, size=9)
    else:
        pos_x_err = np.zeros(9)
        pos_y_err = np.zeros(9)
        
    dithering.set_focal_position_simple(tabled_x_pos[source_id], tabled_y_pos[source_id])
    #if pos_rms > 0:
    #    dithering.place_fiber([pos_x_err, pos_y_err])
    #dithering.place_fiber(pos_x_err[0], pos_y_err[0])
    dithering.set_theta_0(-5.*u.deg)
    dithering.set_phi_0(10.*u.deg)
    SNR    = []
    signal = []
    x      = []
    y      = []

    focal_x = dithering.focal_x[0]
    focal_y = dithering.focal_y[0]

    # initial point
    #if pos_rms > 0:
    #    dithering.place_fiber([x_offset+pos_x_err[0], y_offset+pos_y_err[0]])
    #else:
    #    dithering.place_fiber([x_offset, y_offset])
    dithering.place_fiber([x_offset+pos_x_err[0], y_offset+pos_y_err[0]])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[0], airmass_offset=random_airmass_offsets[0])
    SNR.append(np.median(dithering.SNR['b'][0]))
    signal.append(np.median(dithering.signal['b'][0]))
    x.append(0)
    y.append(0)

    # dithering starts here
    # it is a very simple method to find the maximum point
    # the optimal is with 6 measurements (including the initial measurement)
    # -- first dither
    x_dither = search_radius*np.cos(30.*u.deg)
    y_dither = search_radius*np.sin(30.*u.deg)
    #if pos_rms > 0:
    #    dithering.place_fiber([x_offset+x_dither+pos_x_err[1], y_offset+y_dither+pos_y_err[1]])
    #else:
    #    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
    dithering.place_fiber([x_offset+x_dither+pos_x_err[1], y_offset+y_dither+pos_y_err[1]])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[1], airmass_offset=random_airmass_offsets[1])
    SNR.append(np.median(dithering.SNR['b'][0]))
    signal.append(np.median(dithering.signal['b'][0]))
    x.append(x_dither)
    y.append(y_dither)

    # -- second dither
    x_dither = -1*search_radius*np.cos(30.*u.deg)
    y_dither = search_radius*np.sin(30.*u.deg)
    #if pos_rms > 0:
    #    dithering.place_fiber([x_offset+x_dither+pos_x_err[2], y_offset+y_dither+pos_y_err[2]])
    #else:
    #    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
    dithering.place_fiber([x_offset+x_dither+pos_x_err[2], y_offset+y_dither+pos_y_err[2]])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[2], airmass_offset=random_airmass_offsets[2])
    SNR.append(np.median(dithering.SNR['b'][0]))
    signal.append(np.median(dithering.signal['b'][0]))
    x.append(x_dither)
    y.append(y_dither)
    
    # -- third dither
    x_dither = 0. 
    y_dither = -1*search_radius
    #if pos_rms > 0:
    #    dithering.place_fiber([x_offset+x_dither+pos_x_err[3], y_offset+y_dither+pos_y_err[3]])
    #else:
    #    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
    dithering.place_fiber([x_offset+x_dither+pos_x_err[3], y_offset+y_dither+pos_y_err[3]])
    dithering.run_simulation(source_type, *source, report=False, \
                             seeing_fwhm_ref_offset=random_seeing_offsets[3], airmass_offset=random_airmass_offsets[3])
    SNR.append(np.median(dithering.SNR['b'][0]))
    signal.append(np.median(dithering.signal['b'][0]))
    x.append(x_dither)
    y.append(y_dither)
    
    # -- next two dithering depends on the maximum among the ones searches
    max_idx = np.argmax(SNR)
    new_angles_primary   = check_angles_primary[max_idx]
    for i in range(2):
        x_dither = search_radius*np.cos(new_angles_primary[i]*u.deg)+x[max_idx]
        y_dither = search_radius*np.sin(new_angles_primary[i]*u.deg)+y[max_idx]
        #if pos_rms > 0:
        #    dithering.place_fiber([x_offset+x_dither+pos_x_err[4+i], y_offset+y_dither+pos_y_err[4+i]])
        #else:
        #    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
        dithering.place_fiber([x_offset+x_dither+pos_x_err[4+i], y_offset+y_dither+pos_y_err[4+i]])
        dithering.run_simulation(source_type, *source, report=False, \
                                 seeing_fwhm_ref_offset=random_seeing_offsets[4+i], airmass_offset=random_airmass_offsets[4+i])
        SNR.append(np.median(dithering.SNR['b'][0]))
        signal.append(np.median(dithering.signal['b'][0]))
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
        #if pos_rms > 0:
        #    dithering.place_fiber([x_offset+x_dither+pos_x_err[6+i], y_offset+y_dither+pos_y_err[6+i]])
        #else:
        #    dithering.place_fiber([x_offset+x_dither, y_offset+y_dither])
        dithering.place_fiber([x_offset+x_dither+pos_x_err[6+i], y_offset+y_dither+pos_y_err[6+i]])
        dithering.run_simulation(source_type, *source, report=False, \
                                 seeing_fwhm_ref_offset=random_seeing_offsets[6+i], airmass_offset=random_airmass_offsets[6+i])
        SNR.append(np.median(dithering.SNR['b'][0]))
        signal.append(np.median(dithering.signal['b'][0]))
        x.append(x_dither)
        y.append(y_dither)
        
    return x, y, signal, SNR

def run_simulation_rectangular(dithering, source, source_id, random_seeing_offsets, random_airmass_offsets, pos_rms):
    x_pos_moves = [0, -search_radius, 0,  search_radius, -search_radius, search_radius, -search_radius, 0,   search_radius]
    y_pos_moves = [0, search_radius, search_radius, search_radius, 0,   0,  -search_radius, -search_radius, -search_radius]

    # generate the random offset but not reveal it to user until the end
    x_offset = tabled_x_offsets[source_id]
    y_offset = tabled_y_offsets[source_id]
    if pos_rms > 0:
        pos_x_err = np.random.normal(loc=0, scale=pos_rms, size=9)
        pos_y_err = np.random.normal(loc=0, scale=pos_rms, size=9)
    else:
        pos_x_err = np.zeros(9)
        pos_y_err = np.zeros(9)
        
    dithering.set_focal_position_simple(tabled_x_pos[source_id], tabled_y_pos[source_id])
    dithering.place_fiber([x_offset, y_offset])
    dithering.set_theta_0(-5.*u.deg)
    dithering.set_phi_0(10.*u.deg)
    SNR    = []
    signal = []
    x      = []
    y      = []

    #
    # dithering starts here
    # the positioner follows a rectangular pattern around the starting point
    #
    for i in range(len(x_pos_moves)):
        dithering.place_fiber([x_offset+x_pos_moves[i]+pos_x_err[i], y_offset+y_pos_moves[i]+pos_y_err[i]])
        dithering.run_simulation(source_type, *source, report=False, \
                                 seeing_fwhm_ref_offset=random_seeing_offsets[i], \
                                 airmass_offset=random_airmass_offsets[i])
        SNR.append(np.median(dithering.SNR['b'][0]))
        signal.append(np.median(dithering.signal['b'][0]))
        x.append(x_pos_moves[i]+pos_x_err[i])
        y.append(y_pos_moves[i]+pos_y_err[i])
        
    return x, y, signal, SNR

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

parser = argparse.ArgumentParser(description="Script to find the optimal focal point given a set of parameters")
parser.add_argument("--config",         dest="config", default="config/desi-noblur-nooffset.yaml")
parser.add_argument("--setid",          dest="setid",  default=0, type=int)
parser.add_argument("--step-size",      dest="stepsize",         type=float, required=True, help="Step size for the optimization algorithm")
parser.add_argument("--lookupTable",    dest="LUT",              type=str, required=True, help="The table file to run the analysis")
parser.add_argument("--systematic",     dest="systematicoffset", type=float, default=[0.0, 0.0], nargs=2)
parser.add_argument("--half-light-radius",   dest="half_light_radius", type=float,  default=0.5, help="Half width radius of the sources")
parser.add_argument("--seeing_offsets_rms",  dest="seeing_offsets",    type=float,  default=0.0, help="RMS of the seeing offsets")
parser.add_argument("--airmass_offsets_rms", dest="airmass_offsets",   type=float,  default=0.0, help="RMS of the airmass offsets")
parser.add_argument("--number_of_fibers",    dest="num_sources",       type=int,    default=200, help="Number of fibers to simulate at once")
parser.add_argument("--positioner_rms",      dest="pos_rms",           type=float,  default=0.0, help="RMS of the positioner position")
parser.add_argument("--dithering_pattern",   dest="pattern",           type=int,    default=0,   help="Dithering pattern, 0: triangular, 1: rectangular")
parser.add_argument("--output",              dest="outfname",          type=str,    default="results")
parser.add_argument("--prefix",              dest="prefix",            type=str,    default="" required=False, help="The prefix for the output directory")
parsed_args = parser.parse_args()

config_file       = parsed_args.config
setid             = parsed_args.setid
dithering         = Dithering.dithering(config_file=config_file)
dithering_pure    = Dithering.dithering(config_file="config/desi-noblur-nooffset.yaml")
search_radius     = parsed_args.stepsize
systematic_       = parsed_args.systematicoffset
half_light_radius = parsed_args.half_light_radius
num_sources       = parsed_args.num_sources
seeing_offset_rms = parsed_args.seeing_offsets
airmass_offset_rms= parsed_args.airmass_offsets
pos_rms           = parsed_args.pos_rms
outfname          = parsed_args.outfname
pattern           = parsed_args.pattern
LUT_filename      = parsed_args.LUT
prefix            = parsed_args.prefix

tabled_x_pos  = tabled_values['x_pos'][setid*500:(setid+4)*500]
tabled_y_pos  = tabled_values['y_pos'][setid*500:(setid+4)*500]
tabled_x_offsets = tabled_values['x_offsets'][setid*500:(setid+4)*500]
tabled_y_offsets = tabled_values['y_offsets'][setid*500:(setid+4)*500]
tabled_wlens  = tabled_values["wlens"]
tabled_fluxes = tabled_values["fluxes"]
tabled_mags   = tabled_values["mags"]

num_pointings = 1
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

# generate the random airmass offsets to be introduced
# depending on the arguments given, they will be either all zeros
# or they will have a gaussian distribution with a mean given in the config file
# and sigma provided by David's technical note
# same procedure will be followed for the seeing (using seeing_fwhm_ref)
# since all the measurements of an exposure of 500 fibers would be at the same time,
# there are only 9 offset values for each iteration of the algorithm
if seeing_offset_rms<0.01:
    random_seeing_fwhm_ref_offsets = np.zeros(9)*u.arcsec
else:
    random_seeing_fwhm_ref_offsets = (np.random.lognormal(mean=0., sigma=seeing_offset_rms, size=9)-0.98)*u.arcsec

if airmass_offset_rms<0.01:
    random_airmass_offsets         = np.zeros(9)
else:
    random_airmass_offsets         = np.random.normal(0., airmass_offset_rms, 9)

print(random_seeing_fwhm_ref_offsets)
    
# generate the fits file primary header
prihdr = fits.Header()
prihdr['COMMENT'] = "The config file used is {}.".format(config_file)
prihdr["COMMENT"] = "From a uniform distribution with [0, 30), random pointings were chosen"
prihdr['COMMENT'] = "For each boresight altitude and azimuth, there are {} sources randomly located (-1, 1) degrees of the boresight.".format(num_sources)
prihdr["HLR"]     = half_light_radius
prihdu  = fits.PrimaryHDU(header=prihdr)
hdulist = fits.HDUList([prihdu])

for i in range(num_pointings):
    # define the observing conditions
    boresight_alt = random.uniform(10., 30.)
    boresight_az  = random.uniform(10., 30.)

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
    dither_xs  = np.zeros((num_sources,9))
    dither_ys  = np.zeros((num_sources,9))
    calc_snrs    = np.zeros((num_sources,9))
    calc_signals = np.zeros((num_sources,9))
    calc_signal_r = np.zeros(num_sources)
    calc_signal_b = np.zeros(num_sources)
    calc_signal_z = np.zeros(num_sources)
    known_signal_r = np.zeros(num_sources)
    known_signal_b = np.zeros(num_sources)
    known_signal_z = np.zeros(num_sources)
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
                
        # Generate the source to be observed
        if source_type == "qso" or source_type == "QSO":
            source = dithering.generate_source(disk_fraction=0., bulge_fraction=0., 
                                               half_light_disk=0., half_light_bulge=0.)
        elif source_type == "elg" or source_type == "ELG":
            source = dithering.generate_source(disk_fraction=1., bulge_fraction=0., 
                                               half_light_disk=half_light_radius, half_light_bulge=0.)
        elif source_type == "lrg" or source_type == "LRG":
            source = dithering.generate_source(disk_fraction=0., bulge_fraction=1., 
                                               half_light_disk=0., half_light_bulge=half_light_radius)

        w_in = tabled_wlens[j]
        f_in = tabled_fluxes[j]
        mag  = tabled_mags[j]
        dithering.desi.source.update_in("F type star", "star", w_in*u.angstrom, f_in*1e-17*u.erg/(u.angstrom*u.cm*u.cm*u.s))
        dithering.desi.source.update_out()
        dithering_pure.desi.source.update_in("F type star", "star", w_in*u.angstrom, f_in*1e-17*u.erg/(u.angstrom*u.cm*u.cm*u.s))
        dithering_pure.desi.source.update_out()
        
        dithering_pure.set_focal_position_simple(tabled_x_pos[j], tabled_y_pos[j])
        dithering_pure.set_theta_0(-5.*u.deg)
        dithering_pure.set_phi_0(10.*u.deg)
        dithering_pure.run_simulation(source_type, *source, report=False, \
                                      seeing_fwhm_ref_offset=1e-12*u.arcsec, airmass_offset=1e-12)
        
        #try:
        if True:
            x_offset = tabled_x_offsets[j]
            y_offset = tabled_y_offsets[j]
            focal_x  = tabled_x_pos[j]
            focal_y  = tabled_y_pos[j]

            start_time = time.time()
            if pattern == 1:
                xs, ys, signals, SNRs = run_simulation_rectangular(dithering, source, j, random_seeing_fwhm_ref_offsets, random_airmass_offsets, pos_rms)
            else:
                xs, ys, signals, SNRs = run_simulation_triangulate(dithering, source, j, random_seeing_fwhm_ref_offsets, random_airmass_offsets, pos_rms)
            #print(time.time()-start_time)
            start_time = time.time()
            
            opt_x_offset, opt_y_offset, opt_x_sigma, opt_y_sigma, opt_phi= fit_simulation(xs, ys, signals, SNRs)
            #print(opt_x_offset, opt_y_offset, opt_x_sigma, opt_y_sigma, opt_phi)
            dithering.place_fiber([x_offset+opt_x_offset, y_offset+opt_y_offset])
            dithering.run_simulation(source_type, *source, report=False)
            #print(time.time()-start_time)
            
            focal_xs[j] = focal_x
            focal_ys[j] = focal_y
            known_offset_x[j] = x_offset
            known_offset_y[j] = y_offset
            known_systematic_x[j] = systematic_[0]
            known_systematic_y[j] = systematic_[1]
            magnitudes[j] = np.array(mag)
            
            dither_xs[j]    = np.array(xs)
            dither_ys[j]    = np.array(ys)
            calc_snrs[j]    = np.array(SNRs)
            calc_signals[j] = np.array(signals)
            
            calc_offset_x[j] = opt_x_offset
            calc_offset_y[j] = opt_y_offset
            calc_sigma_x[j] = opt_x_sigma
            calc_sigma_y[j] = opt_y_sigma
                        
            calc_snr_b[j] = (np.median(dithering.SNR['b'][0]))
            calc_snr_r[j] = (np.median(dithering.SNR['r'][0]))
            calc_snr_z[j] = (np.median(dithering.SNR['z'][0]))
            calc_signal_b[j] = (np.median(dithering.signal['b'][0]))
            calc_signal_r[j] = (np.median(dithering.signal['r'][0]))
            calc_signal_z[j] = (np.median(dithering.signal['z'][0]))
            
            known_snr_b[j] = (np.median(dithering_pure.SNR['b'][0]))
            known_snr_r[j] = (np.median(dithering_pure.SNR['r'][0]))
            known_snr_z[j] = (np.median(dithering_pure.SNR['z'][0]))
            known_signal_b[j] = (np.median(dithering_pure.signal['b'][0]))
            known_signal_r[j] = (np.median(dithering_pure.signal['r'][0]))
            known_signal_z[j] = (np.median(dithering_pure.signal['z'][0]))

            del focal_x, focal_y, x_offset, y_offset
            del mag, f_in, w_in
            del xs, ys, SNRs, signals
            del opt_x_offset, opt_y_offset, opt_x_sigma, opt_y_sigma
            del dithering.SNR, dithering.signal
            del dithering_pure.SNR, dithering_pure.signal
            
        #except:
        else:
            #print("problem with the current fiber dithering {}... moving to the next one...".format(j))
            continue
        
    hodor = fits.Header()
    hodor['airmass_offset_rms'] = airmass_offset_rms
    hodor['seeing_offset_rms']  = seeing_offset_rms
    hodor['search_radiuss']     = search_radius
    hodor['positioner_error']   = pos_rms
    hodor['seeing_offset_0']    = random_seeing_fwhm_ref_offsets[0].value
    hodor['seeing_offset_1']    = random_seeing_fwhm_ref_offsets[1].value
    hodor['seeing_offset_2']    = random_seeing_fwhm_ref_offsets[2].value
    hodor['seeing_offset_3']    = random_seeing_fwhm_ref_offsets[3].value
    hodor['seeing_offset_4']    = random_seeing_fwhm_ref_offsets[4].value
    hodor['seeing_offset_5']    = random_seeing_fwhm_ref_offsets[5].value
    hodor['seeing_offset_6']    = random_seeing_fwhm_ref_offsets[6].value
    hodor['seeing_offset_7']    = random_seeing_fwhm_ref_offsets[7].value
    hodor['seeing_offset_8']    = random_seeing_fwhm_ref_offsets[8].value
    
    thdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="boresight_alt",      array=bore_alt, format="E"),
         fits.Column(name="boresight_az",       array=bore_az, format="E"),
         fits.Column(name="source_alt",         array=src_alt, format="E"),
         fits.Column(name="source_az",          array=src_az, format="E"),
         fits.Column(name="known_offset_x",     array=known_offset_x, format="E"),
         fits.Column(name="known_offset_y",     array=known_offset_y, format="E"),
         fits.Column(name="known_systematic_x", array=known_systematic_x, format="E"),
         fits.Column(name="known_systematic_y", array=known_systematic_y, format="E"),
         fits.Column(name="calc_offset_x",      array=calc_offset_x, format="E"),
         fits.Column(name="calc_offset_y",      array=calc_offset_y, format="E"),
         fits.Column(name="calc_sigma_x",       array=calc_sigma_x, format="E"),
         fits.Column(name="calc_sigma_y",       array=calc_sigma_y, format="E"),
         fits.Column(name="known_snr_b",        array=known_snr_b, format="E"),
         fits.Column(name="known_snr_r",        array=known_snr_r, format="E"),
         fits.Column(name="known_snr_z",        array=known_snr_z, format="E"),
         fits.Column(name="calc_snr_b",         array=calc_snr_b, format="E"),
         fits.Column(name="calc_snr_r",         array=calc_snr_r, format="E"),
         fits.Column(name="calc_snr_z",         array=calc_snr_z, format="E"),
         fits.Column(name="calc_snrs",          array=calc_snrs,  format="9E"),
         fits.Column(name="known_signal_b",     array=known_signal_b, format="E"),
         fits.Column(name="known_signal_r",     array=known_signal_r, format="E"),
         fits.Column(name="known_signal_z",     array=known_signal_z, format="E"),
         fits.Column(name="calc_signal_b",      array=calc_signal_b, format="E"),
         fits.Column(name="calc_signal_r",      array=calc_signal_r, format="E"),
         fits.Column(name="calc_signal_z",      array=calc_signal_z, format="E"),
         fits.Column(name="dither_pos_x",       array=dither_xs,  format="9E"),
         fits.Column(name="dither_pos_y",       array=dither_ys,  format="9E"),
         fits.Column(name="calc_signals",       array=calc_signals,  format="9E"),
         fits.Column(name="focal_x",            array=focal_xs, format='E'),
         fits.Column(name="focal_y",            array=focal_ys, format="E"),
         fits.Column(name="mag",                array=magnitudes, format="5E"),
        ], header=hodor)
    hdulist.append(thdu)
    
try:
    os.mkdir("{}/data/".format(prefix))
except:
    print("main folder exists... checking the subfolder...")

try:
    os.mkdir("{}/data/{}um".format(prefix, search_radius))
except:
    print("subfolder exists.. moving on to saving the file")

temp_filename = "{}/data/{}um/{}.fits".format(prefix, search_radius, outfname)
filename = temp_filename
trial = 1
while True:
    if os.path.exists(filename):
        filename = temp_filename.split(".fits")[0]+"_{}".format(trial)+".fits"
        trial = trial + 1
    else:
        break
hdulist.writeto(filename)
