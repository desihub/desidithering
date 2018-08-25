import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import time
import math
import sys

from astropy.io import fits

x = np.linspace(-100, 100, 200)
y = np.linspace(-100, 100, 200)
xv, yv = np.meshgrid(x, y)

def twoDGaussian(x, y, amp, xo, yo, sigma_1, sigma_2, phi):
    A = (np.cos(phi) / sigma_1)**2 + (np.sin(phi) / sigma_2)**2
    B = (np.sin(phi) / sigma_1)**2 + (np.cos(phi) / sigma_2)**2
    C = 2 * np.sin(phi) * np.cos(phi)*(1/sigma_1**2 - 1/sigma_2**2)
    return amp * np.exp( -0.5 * ( A*(x-xo)**2 + B*(y-yo)**2 + C*(x-xo)*(y-yo) ) )

def calculate_integral(x_pos, y_pos, xo, yo, amp, sigma_1):
    max_x = max(x_pos, xo)+100
    min_x = min(x_pos, xo)-100
    max_y = max(y_pos, yo)+100
    min_y = min(y_pos, yo)-100
    xs = np.linspace(min_x, max_x, 400)
    ys = np.linspace(min_y, max_y, 400)
    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    xv, yv = np.meshgrid(xs, ys)
    integral = 0
    for i in range(400):
        for j in range(400):
            integral += integrand(xv[i,j], yv[i,j], x_pos, y_pos, amp, xo, yo, sigma_1)*dx*dy
    return integral

def integrand(x, y, x_offset, y_offset, amp, xo, yo, sigma_1):
    condition1 = (np.sqrt((x-x_offset)**2+(y-y_offset)**2))<=sigma_1
    condition2 = (np.sqrt((x-xo)**2+(y-yo)**2))<=sigma_1
    if condition1 and condition2:
        return twoDGaussian(x, y, amp, xo, yo, sigma_1, sigma_1, 0.0)
    else:
        return 0

class MCMC_fitter:

    def __init__(self, filename, max_fiber=None, debug=False):
        # define a set of constants
        self.fiber_radius = 55.
        # the ones below may not be needed
        self.x    = None
        self.y    = None
        self.yerr = None
        self.filename = filename
        self.debug = debug
        self.max_fiber = max_fiber if max_fiber is not None else 5
        
    def read_data(self, filename=None):
        if filename == None:
            filename = self.filename
        hdus = fits.open(filename)
        data = hdus[1].data
        self.known_offset_x = data['known_offset_x'][:self.max_fiber] # of size num_fibers
        self.known_offset_y = data['known_offset_y'][:self.max_fiber] # of size num_fibers
        self.dither_pos_x = data['dither_pos_x'][:self.max_fiber]     # of size num_fibersx9
        self.dither_pos_y = data['dither_pos_y'][:self.max_fiber]     # of size num_fibersx9
        self.signals = data['calc_signals'][:self.max_fiber]          # of size num_fibersx9
        self.snrs = data['calc_snrs'][:self.max_fiber]                # of size num_fibersx9
        self.est_pos_x = data['calc_offset_x'][:self.max_fiber]       # of size num_fibers
        self.est_pos_y = data['calc_offset_y'][:self.max_fiber]       # of size num_fibers
        self.est_sigma = data['calc_sigma_x'][:self.max_fiber]        # of size num_fibers
        self.num_fibers = len(self.known_offset_x)
        self.num_dithers = len(self.dither_pos_x[0])
        self.num_params = self.num_fibers * 3 + self.num_dithers
        if self.debug:
            print("number of fibers in the simulation: {}".format(self.num_fibers))
            print("number of dithers in the simulations: {}".format(self.num_dithers))
            print("number of unknowns to solve: {}".format(self.num_params))
        
    def lnprior(self, theta):
        # define the priors in the hyperspace
        condition1 = np.less_equal(theta[:9], 90)
        condition2 = np.greater(theta[:9], 20)
        sigma_condition = np.logical_and(condition1, condition2)

        condition1 = np.less_equal(theta[9::3], 100)
        condition2 = np.greater(theta[9::3], 0)
        x_offset_condition = np.logical_and(condition1, condition2)

        condition1 = np.less_equal(theta[10::3], 100)
        condition2 = np.greater(theta[10::3], 0)
        y_offset_condition = np.logical_and(condition1, condition2)

        condition1 = np.less_equal(theta[11::3], 1e-1)
        condition2 = np.greater(theta[11::3], 1e-5)
        amplitude_condition = np.logical_and(condition1, condition2)

        overall_condition = sigma_condition.all() and x_offset_condition.all() and \
            y_offset_condition.all() and amplitude_condition.all()

        if overall_condition:
            return 0.0
        else:
            return -np.inf
        
    # the way this is implemented may not work
    # we won't be passing x, y, yerr values
    # instead they will be set in the function itself
    # for contructing the MCMC, be careful and assign them
    # to None and check whether you get some reasonable
    # results. Otherwise, assign an array with ndim=1 and pass it.
    # And if that does not workm this will be extremely
    # DIFFICULT!!!
    def lnlike(self, theta, x, y, yerr):
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
        for iFiber in range(self.num_fibers):
            for iDither in range(self.num_dithers):
                model_val = calculate_integral(self.dither_pos_x[iFiber, iDither], self.dither_pos_y[iFiber, iDither],
                                               theta[9+iFiber*3], theta[9+iFiber*3+1], theta[9+iFiber*3+2], theta[iDither])
                y = self.signals[iFiber][iDither]
                yerr = (self.snrs[iFiber][iDither] / self.signals[iFiber][iDither])**-1
                #print(iFiber, iDither, model_val, y, yerr, theta[9+iFiber*3], theta[9+iFiber*3+1], theta[9+iFiber*3+2], theta[iDither])
                yerr = yerr if yerr >0. else 1e-10
                chi2 += -0.5*( ( (y-model_val) / yerr )**2. )
        print("Chi2:", chi2)
        return chi2

    def lnprob(self, theta, x, y, yerr):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr)
        
    def create_sampler(self):
        self.ndim     = self.num_params
        self.nwalkers = self.ndim*4
        self.sampler  = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob,
                                              args=(self.x, self.y, self.yerr))

    def run_MCMC(self):
        sigma_min = np.full(shape=9, fill_value=20)
        sigma_max = np.full(shape=9, fill_value=90)
        others_min = np.full(shape=(self.num_fibers, 3), fill_value=[0., 0., 1e-5]).reshape(1, -1)[0]
        others_max = np.full(shape=(self.num_fibers, 3), fill_value=[100., 100., 1e-1]).reshape(1, -1)[0]
        samples_min = np.append(sigma_min, others_min)
        samples_max = np.append(sigma_max, others_max)
        if self.debug:
            print("dimension of the samples_min: {}".format(len(samples_min)))
            print("dimension of the samples_max: {}".format(len(samples_max)))
        samples_size= samples_max - samples_min
        samples     = [samples_min + samples_size * np.random.rand(self.ndim) for i in range(self.nwalkers)]
        self.sampler.run_mcmc(samples, 10)
        self.samples= self.sampler.flatchain
        print(self.samples)
        
    def plot_results(self):
        start_time = time.time()
        corner.corner(self.samples, labels=["param1", "param2", "param3"],
                      range=[[3.0, 5.5], [-1.5, 0.0], [0.0, 2.0]],
                      truths=[self.params[0], self.params[1], self.params[2]])
        print("total time elapsed for plotting the results: {}".format(time.time()-start_time))
        plt.show()

if __name__=="__main__":
    start_time = time.time()
    fitter = MCMC_fitter("example_result.fits", debug=True, max_fiber=2)
    fitter.read_data()
    fitter.create_sampler()
    fitter.run_MCMC()
    print("total time elapsed for MCMC: {}".format(time.time()-start_time))

