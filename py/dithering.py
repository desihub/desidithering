
# This script does dithering for a given source.
# Author: Tolga Yapici
# email: tyapici[at]ur.rochester.edu

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
import random

import time

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

import desimodel.focalplane as fplane

ARCSEC2DEG = 0.000277778
UM2MM      = 0.001

class dithering:
    """
    Initialize a dithering module

    Parameters
    ----------
    config_file : string
        The configuration filename with the parameters for fiber acceptance
        and other related calculations
    """
    def __init__(self, config_file="../config/desi-blur.yaml", verbose=False, output_file=None):
        self.config_file = config_file
        cfg = specsim.config.load_config(self.config_file)
        self.desi        = sim.Simulator(self.config_file, num_fibers=1)
        wavelength       = self.desi.simulated['wavelength']
        wlen_unit        = wavelength.unit
        self.num_wlen    = len(wavelength)
        self.wlen_grid   = np.linspace(wavelength.data[0], wavelength.data[-1], self.num_wlen) * wlen_unit
        self.pointing    = self.desi.observation.pointing
        self.place_fiber([0., 0.])
        self.radius      = 3*u.mm
        self.theta       = 0*u.deg
        self.phi         = 0*u.deg
        self.theta_0     = 0*u.deg
        self.phi_0       = 0*u.deg
        self.prev_theta  = 0*u.deg
        self.prev_phi    = 0*u.deg
        self.angle       = 0
        #self.platescale  = cfg.get_constants(cfg.instrument.plate_scale, ['value'])['value']
        self.verbose     = verbose
        self.output_file = output_file
        if self.output_file is not None:
            f = open(self.output_file, "w")
            f.write("Output generated at : \n")
            f.close()
        # need to transform platescale to the correct units to be used in transformation function
        #self.platescale *= (UM2MM / ARCSEC2DEG)
            
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
    def generate_source(self, disk_fraction=0., bulge_fraction=0., vary='', seed=23, minormajor=[1,1], half_light_disk=0., half_light_bulge=0.):
        gen = np.random.RandomState(seed)
        varied = vary.split(',')
        source_fraction = np.tile([disk_fraction, bulge_fraction], (1, 1))
        source_half_light_radius = np.tile([half_light_disk, half_light_bulge], (1, 1))
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
                       source_minor_major_axis_ratio, source_position_angle, report=True, seeing_fwhm_ref_offset=0*u.arcsec,
                       airmass_offset=0):
        # store the variables in a local variable to revert back at the end
        org_seeing_fwhm_ref = self.desi.atmosphere.seeing_fwhm_ref
        org_airmass         = self.desi.atmosphere.airmass
        self.desi.atmosphere.seeing_fwhm_ref += seeing_fwhm_ref_offset
        self.desi.atmosphere.airmass         += airmass_offset
        self.get_fiber_acceptance_fraction(source_type, source_fraction, source_half_light_radius, source_minor_major_axis_ratio, source_position_angle)
        self.desi.simulate(fiber_acceptance_fraction=self.fiber_acceptance_fraction)
        self.SNR = {}
        self.signal = {}
        for output in self.desi.camera_output:
            snr = (output['num_source_electrons'][:, 0])/np.sqrt(output['variance_electrons'][:, 0])
            signal = (output['num_source_electrons'][:, 0])
            self.SNR[output.meta['name']] = [snr, output.meta['pixel_size']]
            self.signal[output.meta['name']] = [signal, output.meta['pixel_size']]
        if report:
            self.report(simple=False)
        # variables are restored using the local storage variables
        self.desi.atmosphere.seeing_fwhm_ref = org_seeing_fwhm_ref
        self.desi.atmosphere.airmass         = org_airmass
        del org_seeing_fwhm_ref
        del org_airmass
        
    def report(self, simple=True):
        if self.output_file is not None:
            f = open(self.output_file, "a")
            f.write("boresight position is   : {0:.3f} , {1:.3f}\n".format(self.alt_bore, self.az_bore))
            f.write("source position is      : {0:.3f} , {1:.3f}\n".format(self.alt, self.az))
            f.write("fiber position is       : {0:.3f} , {1:.3f}\n".format(self.fiber_x[0], self.fiber_y[0]))
            f.write("focal plane position is : {0:.3f} , {1:.3f}\n".format(self.focal_x[0], self.focal_y[0]))
            f.write("fiber placement         : {0} um, {1} um\n".format(self.fiber_placement[0], self.fiber_placement[1]))
            if not simple:
                f.write("With the current configuration, SNR are:\n")
                for camera_name in self.SNR:
                    f.write("-- camera {0}: {1:.3f} / {2}\n".format(camera_name, np.median(self.SNR[camera_name][0]), self.SNR[camera_name][1]))
            f.close()
        else:
            print("boresight position is   : {0:.3f} , {1:.3f}".format(self.alt_bore, self.az_bore))
            print("source position is      : {0:.3f} , {1:.3f}".format(self.alt, self.az))
            print("fiber position is       : {0:.3f} , {1:.3f}".format(self.fiber_x[0], self.fiber_y[0]))
            print("focal plane position is : {0:.3f} , {1:.3f}".format(self.focal_x[0], self.focal_y[0]))
            print("fiber placement         : {0} um, {1} um".format(self.fiber_placement[0], self.fiber_placement[1]))
            if not simple:
                print("With the current configuration, SNR are:")
                for camera_name in self.SNR:
                    print("-- camera {0}: {1:.3f} / {2}".format(camera_name, np.median(self.SNR[camera_name][0]), self.SNR[camera_name][1]))

    def set_source_position(self, alt, az):
        self.alt = alt
        self.az = az

    def set_boresight_position(self, alt, az):
        self.alt_bore = alt
        self.az_bore = az

    """
    def set_focal_plane_position(self):
        focal_x, focal_y = (trans.altaz_to_focalplane(self.alt, self.az, self.alt_bore, self.az_bore, platescale=self.platescale))
        self.focal_x = [(focal_x.to(u.mm).value)]*u.mm
        self.focal_y = [(focal_y.to(u.mm).value)]*u.mm
        self.fiber_x = self.focal_x
        self.fiber_y = self.focal_y
        self.fiber_placement = [self.focal_x.to(u.um).value-self.fiber_x.to(u.um).value, 
                                self.focal_y.to(u.um).value-self.fiber_y.to(u.um).value]
        self.theta      = 0*u.deg
        self.prev_theta = 0*u.deg
        self.phi        = 0*u.deg
        self.prev_phi   = 0*u.deg
    """
    
    def set_focal_position(self):
        fp  = fplane.FocalPlane()
        fp.set_tele_pointing(self.alt_bore.value, self.az_bore.value)
        pos = fp.radec2xy(self.alt.value, self.az.value)
        self.focal_x = [pos[0]]*u.mm
        self.focal_y = [pos[1]]*u.mm
        self.fiber_x = self.focal_x
        self.fiber_y = self.focal_y
        self.fiber_placement = [self.focal_x.to(u.um).value-self.fiber_x.to(u.um).value, 
                                self.focal_y.to(u.um).value-self.fiber_y.to(u.um).value]
        self.theta      = 0*u.deg
        self.prev_theta = 0*u.deg
        self.phi        = 0*u.deg
        self.prev_phi   = 0*u.deg
        
    def set_focal_position_simple(self, x, y):
        self.focal_x = [x]*u.mm
        self.focal_y = [y]*u.mm
        self.fiber_x = self.focal_x
        self.fiber_y = self.focal_y
        self.fiber_placement = [self.focal_x.to(u.um).value-self.fiber_x.to(u.um).value, 
                                self.focal_y.to(u.um).value-self.fiber_y.to(u.um).value]
        self.theta      = 0*u.deg
        self.prev_theta = 0*u.deg
        self.phi        = 0*u.deg
        self.prev_phi   = 0*u.deg
        
    def check_focal_position(self, check=False):
        radius      = self.desi.instrument.field_radius.to(u.mm).value
        curr_radius = np.sqrt(self.focal_x.to(u.mm).value **2 + self.focal_y.to(u.mm).value **2)
        if check is False:
            return True
        if curr_radius <= radius:
            return True
        else:
            return False
        
    """
    Function to change the boresight position. This is done by the following transformation.
    The transformation may not be correct. Need to consult David Kirkby
    
    Parameters
    ----------
    alt_bore : float
        new boresight altitude angle in degrees (unit needs to be provided)
    az_bore : float
        new boresight azimuth angle in degrees (unit needs to be provided)
    """
    def change_alt_az_bore_position(self, alt_bore, az_bore):
        #platescale   = 70.3947#self.desi.instrument.plate_scale
        prev_focal_x = self.fiber_x
        prev_focal_y = self.fiber_y
        focal_x, focal_y = (trans.altaz_to_focalplane(self.alt, self.az, alt_bore, az_bore, platescale=self.platescale))#*u.mm/u.deg))
        self.fiber_placement = [focal_x.to(u.um).value - self.fiber_x.to(u.um).value, focal_y.to(u.um).value - self.fiber_y.to(u.um).value]
        #[focal_x.to(u.um).value - prev_focal_x.to(u.um).value, focal_y.to(u.um).value - prev_focal_y.to(u.um).value]
        self.focal_x = [(focal_x.to(u.mm).value)]*u.mm
        self.focal_y = [(focal_y.to(u.mm).value)]*u.mm
        self.set_boresight_position(alt_bore, az_bore)

    """
    Function to rotate the position on the 2nd axis. The rotation is done for $\Phi$ only for now
    
    Parameters
    ----------
    angle : float
        rotation angle in degrees (unit needs to be provided)
    """ 
    def rotate_positioner_r2(self, angle):
        radius = 3*u.mm
        angle  = angle.to(u.rad).value
        delta_x = (radius * np.cos(self.angle)).to(u.um) - (radius * np.cos(self.angle+angle)).to(u.um)
        delta_y = (radius * np.sin(self.angle)).to(u.um) - (radius * np.sin(self.angle+angle)).to(u.um)
        self.angle   = self.angle + angle
        self.fiber_x = self.fiber_x + delta_x
        self.fiber_y = self.fiber_y + delta_y
        self.fiber_placement = [self.fiber_placement[0]-delta_x.value, self.fiber_placement[1]-delta_y.value]

    """
    Function to rotate the position on the 1st axis. The rotation is done for $\Theta$ only for now
    
    Parameters
    ----------
    angle : float
        rotation angle in degrees (unit needs to be provided)
    """ 
    def rotate_positioner_r1(self, angle):
        radius = 3*u.mm
        angle  = angle.to(u.rad).value
        delta_x = (radius * np.cos(self.angle)).to(u.um) - (radius * np.cos(self.angle+angle)).to(u.um)
        delta_y = (radius * np.sin(self.angle)).to(u.um) - (radius * np.sin(self.angle+angle)).to(u.um)
        self.angle   = self.angle + angle
        self.fiber_x = self.fiber_x + delta_x
        self.fiber_y = self.fiber_y + delta_y
        self.fiber_placement = [self.fiber_placement[0]-delta_x.value, self.fiber_placement[1]-delta_y.value]

    def set_positioner_theta(self, angle):
        self.prev_theta = self.theta
        self.theta      = angle
        if self.verbose:
            print("[INFO] positioner theta was {0}".format(self.prev_theta))
            print("[INFO] positioner theta changed to {0}".format(self.theta))
        
    def set_positioner_phi(self, angle):
        self.prev_phi = self.phi
        self.phi      = angle

    def make_positioner_rotation(self):
        prev_x = self.radius * np.cos(self.prev_theta.to(u.rad).value + self.theta_0.to(u.rad).value) + \
                 self.radius * np.cos(self.prev_phi.to(u.rad).value + self.phi_0.to(u.rad).value)
        x      = self.radius * np.cos(self.theta.to(u.rad).value + self.theta_0.to(u.rad).value) + \
                 self.radius * np.cos(self.phi.to(u.rad).value + self.phi_0.to(u.rad).value)
        prev_y = self.radius * np.sin(self.prev_theta.to(u.rad).value + self.theta_0.to(u.rad).value) + \
                 self.radius * np.sin(self.prev_phi.to(u.rad).value + self.phi_0.to(u.rad).value)
        y      = self.radius * np.sin(self.theta.to(u.rad).value + self.theta_0.to(u.rad).value) + \
                 self.radius * np.sin(self.phi.to(u.rad).value + self.phi_0.to(u.rad).value)
        delta_x = (x - prev_x).to(u.um)
        delta_y = (y - prev_y).to(u.um)
        self.fiber_x = self.fiber_x + delta_x
        self.fiber_y = self.fiber_y + delta_y
        #self.fiber_placement = [self.fiber_placement[0]+delta_x.value, self.fiber_placement[1]+delta_y.value]
        self.fiber_placement = [self.focal_x.to(u.um).value-self.fiber_x.to(u.um).value, 
                                self.focal_y.to(u.um).value-self.fiber_y.to(u.um).value]
        return

    def add_random_offset_fiber_position(self, mean_x=0*u.um, var_x=1*u.um, mean_y=0*u.um, var_y=1*u.um):
        random_offset_x = random.gauss(mean_x, var_x)
        random_offset_y = random.gauss(mean_y, var_y)
        self.fiber_x = self.fiber_x + random_offset_x
        self.fiber_y = self.fiber_y + random_offset_y
        self.fiber_placement = [self.fiber_placement[0] + random_offset_x.to(u.um).value,
                                self.fiber_placement[1] + random_offset_y.to(u.um).value]

    def add_random_boresight_offset(self, mean_alt=0*u.arcsec, var_alt=1*u.arcsec, mean_az=0*u.arcsec, var_az=1*u.arcsec):
        random_offset_alt = random.gauss(mean_alt, var_alt)
        random_offset_az  = random.gauss(mean_az, var_az)
        self.alt_bore = self.alt_bore + random_offset_alt
        self.az_bore  = self.az_bore + random_offset_az
        self.change_alt_az_bore_position(self.alt_bore, self.az_bore)

    def set_theta_0(self, angle):
        self.theta_0 = angle

    def set_phi_0(self, angle):
        self.phi_0 = angle
