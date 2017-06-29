
# This script does dithering for a given source.
# Author: Tolga Yapici
# email: tyapici[at]pa.msu.edu

#
# TODO:
# 1. rotate the fiber along an axis mimicing the positioners
# 2. ???

# general packages
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.table
import scipy

# specsim (DESI related packages)
# some of them are probably not being used in this module
# these will be cleaned up later
import specsim
import specsim.quickspecsim as qspecsim
import specsim.instrument as inst
import specsim.source as src
import specsim.config as conf
import specsim.fiberloss as floss
import specsim.observation as obs
import specsim.simulator as sim
import specsim.atmosphere as atm
import specsim.transform as trans

class dithering:
    """
    Initialize a dithering module

    Parameters
    ----------
    config_file : string
        The configuration filename with the parameters for fiber acceptance
        and other related calculations
    """
    def __init__(self, config_file="./config/desi-blur-offset.yaml"):
        self.config_file = config_file
        self.desi        = sim.Simulator(self.config_file, num_fibers=1)
        wavelength       = self.desi.simulated['wavelength']
        wlen_unit        = wavelength.unit
        self.num_wlen    = len(wavelength)
        self.wlen_grid   = np.linspace(wavelength.data[0], wavelength.data[-1], self.num_wlen) * wlen_unit
        self.pointing    = self.desi.observation.pointing
        self.place_fiber([0., 0.])

    """
    Function to generate a single source
    Function is only for generating the source profile parameters
    it does not have anything about the source type explicitly
    the default profile is a point source QSO

    Parameters
    ----------
    disk_fraction : 
    
    bulge_fraction :
    
    vary : 
    
    minormajor : 
    
    Returns
    -------
    source_fraction :
    
    source_half_light_radius :
    
    source_minor_major_axis : 
    
    source_position_angle : 

    """
    def generate_source(self, disk_fraction=0., bulge_fraction=0., vary='', seed=23, minormajor=[1,1]):
        gen = np.random.RandomState(seed)
        varied = vary.split(',')
        source_fraction = np.tile([disk_fraction, bulge_fraction], (1, 1))
        source_half_light_radius = np.tile([0.45, 1.0], (1, 1))
        source_minor_major_axis_ratio = np.tile(minormajor, (1, 1))
        if 'pa' in varied:
            source_position_angle = 360. * gen.uniform(size=(1, 2))
        else:
            source_position_angle = np.tile([0., 0.], (1, 1))
        return source_fraction, source_half_light_radius, source_minor_major_axis_ratio, source_position_angle

    """
    Function to place a fiber on focal plane
    when calculating the fiber acceptance fraction, this becomes important
    
    Parameters
    ----------
    fiber_placement: array
        position of the fiber with respect to focal x and focal y

    Returns
    -------
    None
    """
    def place_fiber(self, fiber_placement):
        self.fiber_placement = fiber_placement

    """
    Function to create the object with galsim parameters
    This function is probably not needed and possible be removed
    That is why no further information is provded here
    """
    def calculateFiberLoss(self, num_pixels=16, oversampling=32, moffat_beta=3.5):
        self.num_pixels   = num_pixels
        self.oversampling = oversampling
        self.moffat_beta  = moffat_beta
        self.fiberloss_calculator = floss.GalsimFiberlossCalculator(self.desi.instrument.fiber_diameter.to(u.um).value,
                                                                    self.wlen_grid.to(u.Angstrom).value,
                                                                    num_pixels=self.num_pixels, oversampling=self.oversampling,
                                                                    moffat_beta=self.moffat_beta,
                                                                    fiber_placement=self.fiber_placement)
        
    """
    Function to return the source type dependent desi-model fiber loss
    
    Parameters
    ----------
    obj_type : string
        source type: 'qso', 'elg', 'lrg'
    Returns
    -------
    t : table
        table with the expected fiber acceptance for a given source type
    """
    def get_desimodel_fiberloss(self, obj_type='qso'):
        path = os.path.join(os.environ['DESIMODEL'], 'data', 'throughput',
                            'fiberloss-{0}.dat'.format(obj_type))
        t = astropy.table.Table.read(path, format='ascii', names=['wlen', 'accept'])
        return t

    """
    Function to evaluate the fiber acceptance given the parameter set
    This function is probably not needed and possible be removed
    That is why no further information is provded here
    """
    def get_fiberloss(self, source_fraction, source_half_light_radius,
                      source_minor_major_axis_ratio, source_position_angle,
                      seeing=1.1*u.arcsec, images_filename=None):
        # Subtract instrumental PSF.
        Jacoby = 0.219
        seeing = 2.35482 * np.sqrt((seeing.to(u.arcsec).value/2.35482) ** 2 - Jacoby**2) * u.arcsec
        # Tabulate seeing.
        self.desi.atmosphere.seeing_fwhm_ref = seeing
        seeing_fwhm = self.desi.atmosphere.get_seeing_fwhm(self.wlen_grid).to(u.arcsec).value
        # Calculate optics.
        scale, blur, offset = self.desi.instrument.get_focal_plane_optics(self.focal_x, self.focal_y, self.wlen_grid)
        # Do the fiberloss calculations.
        return self.fiberloss_calculator.calculate(seeing_fwhm, scale.to(u.um / u.arcsec).value, offset.to(u.um).value,
                                                   blur.to(u.um).value, source_fraction, source_half_light_radius,
                                                   source_minor_major_axis_ratio, source_position_angle,
                                                   saved_images_file=None)

    """
    Function to generate a random position for a fiber
    This function is not probably going to be too useful once we one to check the real source positions and 
    fiber assignments. This is just a placeholder for now just in case some wants to play with random fibers
    
    Parameters
    ----------
    seed : int
        seed for random number generator
    
    Returns
    -------
    None
    but sets up some of the variables of the class
    """
    def generate_fiber_positions(self, seed=123):
        gen = np.random.RandomState(seed)
        focal_r = (np.sqrt(gen.uniform(size=1)) * self.desi.instrument.field_radius)
        phi = 2 * np.pi * gen.uniform(size=1)
        # focal x and y location calculated here
        self.focal_x = np.cos(phi) * focal_r
        self.focal_y = np.sin(phi) * focal_r
        # focal x and y location is stored in variables called "original"
        # so that they can be used later for dithering
        # when the telescope is tilted, "original" focal location will be
        # used to calculate the fiber placement with respect to new focal x and y location
        self.original_focal_x = self.focal_x
        self.original_focal_y = self.focal_y

    """
    Function to get fiber acceptance fraction as a function of wavelength given the parameters
    fiber_acceptance_fraction is later used in the simulation for getting the SNR and some other
    variables
    
    Parameters
    ----------
    source_type : string

    source_fraction : array

    source_half_light_radius : array

    source_minor_major_axis_ratio : array

    source_position_angle : array

    
    Returns
    -------
    None but generates self.fiber_acceptance_fraction
    """
    def get_fiber_acceptance_fraction(self, source_type, source_fraction, source_half_light_radius,
                                      source_minor_major_axis_ratio, source_position_angle):
        self.fiber_acceptance_fraction = floss.calculate_fiber_acceptance_fraction(self.focal_x, self.focal_y, self.wlen_grid,
                                                                                   self.desi.source, self.desi.atmosphere, self.desi.instrument,
                                                                                   source_type, source_fraction, source_half_light_radius,
                                                                                   source_minor_major_axis_ratio,
                                                                                   source_position_angle, fiber_placement=self.fiber_placement)


    """
    Function to run the simulation with all the parameters defined/given

    Parameters
    ----------
    source_type : string

    source_fraction : array

    source_half_light_radius : array

    source_minor_major_axis_ratio : array

    source_position_angle : array
    
    report : bool
        Prints out a report about the results

    """
    def run_simulation(self, source_type, source_fraction, source_half_light_radius,
                       source_minor_major_axis_ratio, source_position_angle, report=True):
        self.get_fiber_acceptance_fraction(source_type, source_fraction, source_half_light_radius, source_minor_major_axis_ratio, source_position_angle)
        self.desi.simulate(fiber_acceptance_fraction=self.fiber_acceptance_fraction)
        self.SNR = {}
        for output in self.desi.camera_output:
            snr = (output['num_source_electrons'][:, 0]/
                   np.sqrt(output['variance_electrons'][:, 0]))
            self.SNR[output.meta['name']] = [snr, output.meta['pixel_size']]
        if report:
            self.report()
        
    def report(self):
        print("Current boresight position is: {0:.3f} , {1:.3f}".format(self.alt_bore, self.az_bore))
        print("Current source position is: {0:.3f} , {1:.3f}".format(self.alt, self.az))
        print("A fiber placement offset of {0} um is added to simuation".format(self.fiber_placement))
        print("With the current configuration, SNR are:")
        for camera_name in self.SNR:
            print("-- camera {0}: {1:.3f} / {2}".format(camera_name, np.median(self.SNR[camera_name][0]), self.SNR[camera_name][1]))

    def set_source_position(self, alt, az):
        self.alt = alt
        self.az = az

    def set_boresight_position(self, alt, az):
        self.alt_bore = alt
        self.az_bore = az

    def set_focal_plane_position(self):
        focal_x, focal_y = (trans.altaz_to_focalplane(self.alt, self.az, self.alt_bore, self.az_bore, platescale=1*u.mm/u.deg))
        self.focal_x = [(focal_x.to(u.mm).value)]*u.mm
        self.focal_y = [(focal_y.to(u.mm).value)]*u.mm
        self.fiber_x = self.focal_x
        self.fiber_y = self.focal_y

    def change_alt_az_bore_position(self, alt_bore, az_bore):
        prev_focal_x = self.fiber_x
        prev_focal_y = self.fiber_y
        focal_x, focal_y = (trans.altaz_to_focalplane(self.alt, self.az, alt_bore, az_bore, platescale=1*u.mm/u.deg))
        self.fiber_placement = [focal_x.to(u.um).value - prev_focal_x.to(u.um).value, focal_y.to(u.um).value - prev_focal_y.to(u.um).value]
        self.focal_x = [(focal_x.to(u.mm).value)]*u.mm
        self.focal_y = [(focal_y.to(u.mm).value)]*u.mm
