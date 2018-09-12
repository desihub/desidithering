
import argparse
import time

import matplotlib as mpl
mpl.use("agg")

import numpy
import matplotlib.pyplot as plt
import emcee
import corner
import math
import sys
import pickle

from multiprocessing import Pool
from astropy.io import fits
#from emcee.utils import MPIPool

#from pathos.multiprocessing import Pool

debug = True

def lnprior(theta):
    # define the priors in the hyperspace
    condition1 = numpy.less_equal(theta[:num_dithers], 90)
    condition2 = numpy.greater(theta[:num_dithers], 20)
    sigma_condition = numpy.logical_and(condition1, condition2)
    
    condition1 = numpy.less_equal(theta[num_dithers::3], 100)
    condition2 = numpy.greater(theta[num_dithers::3], -100)
    x_offset_condition = numpy.logical_and(condition1, condition2)
    
    condition1 = numpy.less_equal(theta[num_dithers+1::3], 100)
    condition2 = numpy.greater(theta[num_dithers+1::3], -100)
    y_offset_condition = numpy.logical_and(condition1, condition2)
    
    condition1 = numpy.less_equal(theta[num_dithers+2::3], 1e-1)
    condition2 = numpy.greater(theta[num_dithers+2::3], 1e-5)
    amplitude_condition = numpy.logical_and(condition1, condition2)
    
    overall_condition = sigma_condition.all() and x_offset_condition.all() and \
        y_offset_condition.all() and amplitude_condition.all()
    
    if overall_condition:
        return 0.0
    else:
        return -numpy.inf

def lnlike(theta, x, y, yerr):
    # since sigma[j] is linked between all fibers,
    # the minimizer needs to work on all samples at once.
    #
    # theta = [x_offset[i], y_offset[i], A[i], sigma[j]] 
    #
    # x    : dither_pos_x[i,j], dither_pos_y[i,j]
    # y    : signal[i,j]
    # i-> fiber number (ndim: 5000)
    # j-> dither number (ndim: 9)
    chi2 = 0
    for iFiber in range(num_fibers):
        for iDither in range(num_dithers):
            model_val = calculate_integral(dither_pos_x[iFiber, iDither], dither_pos_y[iFiber, iDither],
                                           theta[9+iFiber*3], theta[9+iFiber*3+1], theta[9+iFiber*3+2], theta[iDither])
            y = signals[iFiber][iDither]
            yerr = (snrs[iFiber][iDither] / signals[iFiber][iDither])**-1
            yerr = yerr if yerr >0. else 1e-10
            chi2 += -0.5*( ( (y-model_val) / yerr )**2. )
    return chi2

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not numpy.isfinite(lp):
        return -numpy.inf
    return lp + lnlike(theta, x, y, yerr)
        
def create_sampler():
    pool = Pool()
    #if not pool.is_master():
    #    pool.wait()
    #    sys.exit(0)

    ndim     = num_params
    nwalkers = ndim*2
    sampler  = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                              args=(x, y, yerr), pool=pool)#threads=6) #pool=Pool()
    return sampler
    
def run_MCMC(sampler):
    ndim     = num_params
    nwalkers = ndim*2
    sigma_min = numpy.full(shape=num_dithers, fill_value=20)
    sigma_max = numpy.full(shape=num_dithers, fill_value=90)
    others_min = numpy.full(shape=(num_fibers, 3), fill_value=[-100., -100., 1e-5]).reshape(1, -1)[0]
    others_max = numpy.full(shape=(num_fibers, 3), fill_value=[ 100.,  100., 1e-1]).reshape(1, -1)[0]
    samples_min = numpy.append(sigma_min, others_min)
    samples_max = numpy.append(sigma_max, others_max)
    if debug:
        print("dimension of the samples_min: {}".format(len(samples_min)))
        print("dimension of the samples_max: {}".format(len(samples_max)))
    samples_size= samples_max - samples_min
    samples     = [samples_min + samples_size * numpy.random.rand(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(samples, 200)
    #for i, result in enumerate(sampler.run_mcmc(samples, N=nsteps)):
    #    print("{0:5.1%}".format(float(i)/nsteps))
    samples= sampler.flatchain
    return samples
    
def plot_results():
    start_time = time.time()
    corner.corner(samples,
                  labels=["PSF sigma_1", "PSF sigma_2", "PSF sigma_3",
                          "PSF sigma_4", "PSF sigma_5", "PSF sigma_6",
                          "PSF sigma_7", "PSF sigma_8", "PSF sigma_9",
                          "x_offset", "y_offset", "Amplitude"])
    #truths=[params[0], params[1], params[2]])
    print("total time elapsed for plotting the results: {}".format(time.time()-start_time))
    plt.savefig("result_{}.pdf".format(fibers[0]))

def save_results(samples):
    start_time = time.time()
    try:
        results_filename = "result_{}.pkl".format(fibers[0])
        with open(results_filename, "wb") as f:
            pickle.dump(samples, f)
    except:
        results_filename = "resultwith3.pkl".format(fibers[0])
        with open(results_filename, "wb") as f:
            pickle.dump(samples, f)
    print("total time elapsed for saving the results: {}".format(time.time()-start_time))
        
def twoDGaussian(x, y, amp, xo, yo, sigma_1, sigma_2, phi):
    A = (numpy.cos(phi) / sigma_1)**2 + (numpy.sin(phi) / sigma_2)**2
    B = (numpy.sin(phi) / sigma_1)**2 + (numpy.cos(phi) / sigma_2)**2
    C = 2 * numpy.sin(phi) * numpy.cos(phi)*(1/sigma_1**2 - 1/sigma_2**2)
    return amp * numpy.exp( -0.5 * ( A*(x-xo)**2 + B*(y-yo)**2 + C*(x-xo)*(y-yo) ) )

def calculate_integral(x_pos, y_pos, xo, yo, amp, sigma_1):
    max_x = max(x_pos, xo)+100
    min_x = min(x_pos, xo)-100
    max_y = max(y_pos, yo)+100
    min_y = min(y_pos, yo)-100
    num_partitions = 400
    xs = numpy.linspace(min_x, max_x, num_partitions)
    ys = numpy.linspace(min_y, max_y, num_partitions)
    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    xv, yv = numpy.meshgrid(xs, ys)
    integral = 0
    for i in range(num_partitions):
        for j in range(num_partitions):
            integral += integrand(xv[i,j], yv[i,j], x_pos, y_pos, amp, xo, yo, sigma_1)*dx*dy
    return integral

def integrand(x, y, x_offset, y_offset, amp, xo, yo, sigma_1):
    condition1 = (numpy.sqrt((x-x_offset)**2+(y-y_offset)**2))<=sigma_1
    condition2 = (numpy.sqrt((x-xo)**2+(y-yo)**2))<=sigma_1
    if condition1 and condition2:
        return twoDGaussian(x, y, amp, xo, yo, sigma_1, sigma_1, 0.0)
    else:
        return 0


num_dithers = None
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Script to run MCMC fitter on the simulated data")
    parser.add_argument("--filename", dest="filename", default="example_result.fits")
    parser.add_argument("--fiber-no", dest="fiber",    default=0, type=int)
    parsed_args = parser.parse_args()

    start_time = time.time()
    filename = parsed_args.filename
    fibers = [parsed_args.fiber]

    fiber_radius = 55.
    # the ones below may not be needed
    x    = None
    y    = None
    yerr = None
    fibers = fibers if fibers is not None else [1, 2]
    
    hdus = fits.open(filename)
    data = hdus[1].data
    known_offset_x = data['known_offset_x'][fibers] # of size num_fibers
    known_offset_y = data['known_offset_y'][fibers] # of size num_fibers
    dither_pos_x = data['dither_pos_x'][fibers]     # of size num_fibersx9
    dither_pos_y = data['dither_pos_y'][fibers]     # of size num_fibersx9
    signals = data['calc_signals'][fibers]          # of size num_fibersx9
    snrs = data['calc_snrs'][fibers]                # of size num_fibersx9
    est_pos_x = data['calc_offset_x'][fibers]       # of size num_fibers
    est_pos_y = data['calc_offset_y'][fibers]       # of size num_fibers
    est_sigma = data['calc_sigma_x'][fibers]        # of size num_fibers
    num_fibers = len(fibers)
    num_dithers = len(dither_pos_x[0])
    num_params = num_fibers * 3 + num_dithers
    if debug:
        print("number of fibers in the simulation: {}".format(num_fibers))
        print("number of dithers in the simulations: {}".format(num_dithers))
        print("number of unknowns to solve: {}".format(num_params))

    sampler = create_sampler()
    print("sampler generated")
    samples = run_MCMC(sampler)
    print("total time elapsed for MCMC: {}".format(time.time()-start_time))
    save_results(samples)
    plot_results(samples)
    
