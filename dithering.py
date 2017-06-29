
# This script does dithering for a given source.
# Author: Tolga Yapici
# email: tyapici[at]pa.msu.edu

# general packages
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.table

import scipy

# specsim (DESI related packages)
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

# TODO:
#  1. transform package will be used later for source and boring positions
#  2. ...
#

class dithering:

    def __init__(self, telescope_az=0., telescope_lat=0.,
                 source_az=0., source_lat=0.,
                 config_file="./config/desi-blur-offset.yaml"):
        self.config_file = config_file
        self.desi        = sim.Simulator(self.config_file, num_fibers=1)
        wavelength       = self.desi.simulated['wavelength']
        wlen_unit        = wavelength.unit
        self.num_wlen    = len(wavelength)
        self.wlen_grid   = np.linspace(wavelength.data[0], wavelength.data[-1], self.num_wlen) * wlen_unit
        self.pointing    = self.desi.observation.pointing
        self.place_fiber([0., 0.])
        
    # function to generate a single source
    # function is only for generating the source profile parameters
    # it does not have anything about the source type
    # source type will be defined/assigned later
    # the default profile is a point source
    # FINISHED
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

    # function to place a fiber on focal plane
    # when calculating the fiber acceptance fraction, this becomes important
    # FINISHED
    def place_fiber(self, fiber_placement):
        self.fiber_placement = fiber_placement
    
    # function to create the object with galsim parameters
    # FINISHED
    def calculateFiberLoss(self, num_pixels=16, oversampling=32, moffat_beta=3.5):
        self.num_pixels   = num_pixels
        self.oversampling = oversampling
        self.moffat_beta  = moffat_beta
        self.fiberloss_calculator = floss.GalsimFiberlossCalculator(self.desi.instrument.fiber_diameter.to(u.um).value,
                                                                    self.wlen_grid.to(u.Angstrom).value,
                                                                    num_pixels=self.num_pixels, oversampling=self.oversampling,
                                                                    moffat_beta=self.moffat_beta,
                                                                    fiber_placement=self.fiber_placement)
        
    # function to return the source type dependent desi-model fiber loss
    # FINISHED
    def get_desimodel_fiberloss(self, obj_type='qso'):
        path = os.path.join(os.environ['DESIMODEL'], 'data', 'throughput',
                            'fiberloss-{0}.dat'.format(obj_type))
        t = astropy.table.Table.read(path, format='ascii', names=['wlen', 'accept'])
        return t

    # ...
    # FINISHED
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

    # ...
    # FINISHED
    def generate_fiber_positions(self, nfiber=5000, seed=123):
        gen = np.random.RandomState(seed)
        focal_r = (np.sqrt(gen.uniform(size=nfiber)) * self.desi.instrument.field_radius)
        phi = 2 * np.pi * gen.uniform(size=nfiber)
        # focal x and y location calculated here
        self.focal_x = np.cos(phi) * focal_r
        self.focal_y = np.sin(phi) * focal_r
        # focal x and y location is stored in variables called "original"
        # so that they can be used later for dithering
        # when the telescope is tilted, "original" focal location will be
        # used to calculate the fiber placement with respect to new focal x and y location
        self.original_focal_x = self.focal_x
        self.original_focal_y = self.focal_y

    def get_fiber_acceptance_fraction(self, source_type, source_fraction, source_half_light_radius,
                                      source_minor_major_axis_ratio, source_position_angle):
        self.fiber_acceptance_fraction = floss.calculate_fiber_acceptance_fraction(self.focal_x, self.focal_y, self.wlen_grid,
                                                                                   self.desi.source, self.desi.atmosphere, self.desi.instrument,
                                                                                   source_type, source_fraction, source_half_light_radius,
                                                                                   source_minor_major_axis_ratio,
                                                                                   source_position_angle, fiber_placement=self.fiber_placement)

    def run_simulation(self, source_type, source_fraction, source_half_light_radius,
                       source_minor_major_axis_ratio, source_position_angle, report=True):
        self.get_fiber_acceptance_fraction(source_type, source_fraction, source_half_light_radius, source_minor_major_axis_ratio, source_position_angle)
        self.desi.simulate(fiber_acceptance_fraction=self.fiber_acceptance_fraction)
        if report:
            self.report()
        
    def report(self):
        for output in self.desi.camera_output:
            snr = (output['num_source_electrons'][:, 0]/
                   np.sqrt(output['variance_electrons'][:, 0]))
            print("-- camera {0}: {1:.3f} / {2}".format(output.meta['name'], np.median(snr), output.meta['pixel_size']))

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

    def change_alt_az_bore_position(self, alt_bore, az_bore):
        prev_focal_x = self.focal_x
        prev_focal_y = self.focal_y
        focal_x, focal_y = (trans.altaz_to_focalplane(self.alt, self.az, alt_bore, az_bore, platescale=1*u.mm/u.deg))
        self.fiber_placement = [focal_x.to(u.um).value - prev_focal_x.to(u.um).value, focal_y.to(u.um).value - prev_focal_y.to(u.um).value]
        self.focal_x = [(focal_x.to(u.mm).value)]*u.mm
        self.focal_y = [(focal_y.to(u.mm).value)]*u.mm

"""        
dit = dithering()
dit.set_source_position(20.*u.deg, 25.*u.deg)
dit.set_boresight_position(20.*u.deg, 24.5*u.deg)
dit.get_focal_plane_position()
source = dit.generate_source()
dit.place_fiber([0., 0.])
t = dit.get_desimodel_fiberloss('qso')
plt.plot(t['wlen'], t['accept'], 'r--', lw=2, label='desimodel')
# first put the fiber in the middle with no dithering
dit.calculateFiberLoss()
fiber_loss = dit.get_fiberloss(*source)
plt.plot(dit.wlen_grid, fiber_loss.flatten(), label="$\Delta$x=0 um")
print(dit.fiber_placement)
print(dit.focal_x, dit.focal_y)
dit.run_simulation('qso', *source, report=True)

dit.change_alt_az_bore_position(20.*u.deg, 24.45*u.deg)
print(dit.fiber_placement)
print(dit.focal_x, dit.focal_y)
dit.run_simulation('qso', *source, report=True)
"""



"""
fiber_acceptance_fraction = floss.calculate_fiber_acceptance_fraction(dit.focal_x, dit.focal_y,
                                                                      dit.wlen_grid, dit.desi.source, dit.desi.atmosphere,
                                                                      dit.desi.instrument, 'qso', *source)
dit.desi.simulate(fiber_acceptance_fraction=fiber_acceptance_fraction)
print("no movement")
for output in dit.desi.camera_output:
    snr = (output['num_source_electrons'][:, 0]/
           np.sqrt(output['variance_electrons'][:, 0]))
    print("-- camera {0}: {1:.3f} / {2}".format(output.meta['name'], np.median(snr), output.meta['pixel_size']))


fiber_acceptance_fraction = floss.calculate_fiber_acceptance_fraction(dit.focal_x, dit.focal_y,
                                                                      dit.wlen_grid, dit.desi.source, dit.desi.atmosphere,
                                                                      dit.desi.instrument, 'qso', *source, fiber_placement=[0., 5.])
dit.desi.simulate(fiber_acceptance_fraction=fiber_acceptance_fraction)
print("some movement")
for output in dit.desi.camera_output:
    snr = (output['num_source_electrons'][:, 0]/
           np.sqrt(output['variance_electrons'][:, 0]))
    print("-- camera {0}: {1:.3f} / {2}".format(output.meta['name'], np.median(snr), output.meta['pixel_size']))


fiber_acceptance_fraction = floss.calculate_fiber_acceptance_fraction(dit.focal_x, dit.focal_y,
                                                                      dit.wlen_grid, dit.desi.source, dit.desi.atmosphere,
                                                                      dit.desi.instrument, 'qso', *source, fiber_placement=[0., 120.])
dit.desi.simulate(fiber_acceptance_fraction=fiber_acceptance_fraction)
print("lots of movement")
for output in dit.desi.camera_output:
    snr = (output['num_source_electrons'][:, 0]/
           np.sqrt(output['variance_electrons'][:, 0]))
    print("-- camera {0}: {1:.3f} / {2}".format(output.meta['name'], np.median(snr), output.meta['pixel_size']))



interpolator = scipy.interpolate.interp1d(dit.wlen_grid.value, fiber_loss, kind='linear', axis=1,
                                          copy=False, assume_sorted=True)
print(interpolator(dit.wlen_grid).flatten())
print(interpolator(dit.wlen_grid).flatten().shape)
print(dit.wlen_grid)
# Both wavelength grids have the same units, by construction, so no                                                                                      
# conversion factor is required here.
fiber_acceptance_fraction = np.array( interpolator(dit.wlen_grid) )
print(fiber_acceptance_fraction)
print(type(fiber_acceptance_fraction))
dit.desi.simulate(fiber_acceptance_fraction=[fiber_acceptance_fraction]) #np.array(interpolator(dit.wlen_grid).flatten()))
print("no movement")
for output in dit.desi.camera_output:
    snr = (output['num_source_electrons'][:, 0]/
           np.sqrt(output['variance_electrons'][:, 0]))
    print("-- camera {0}: {1:.3f} / {2}".format(output.meta['name'], np.median(snr), output.meta['pixel_size']))
# then start moving the fiber until all the fiber is out
# we only move the fiber in x-direction here in increments of 15 um
for i in range(15, 120, 15):
    dit.place_fiber([i, 0])
    dit.calculateFiberLoss()
    fiber_loss = dit.get_fiberloss(*source)
    plt.plot(dit.wlen_grid, fiber_loss.flatten(), label="$\Delta$x=%d um"%i)
    dit.desi.simulate(fiber_acceptance_fraction=fiber_loss)
    print("delta-x: ", i)
    for output in dit.desi.camera_output:
        snr = (output['num_source_electrons'][:, 0]/
               np.sqrt(output['variance_electrons'][:, 0]))
        print("-- camera {0}: {1:.3f} / {2}".format(output.meta['name'], np.median(snr), output.meta['pixel_size']))
legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Wavelength")
plt.ylabel("Acceptance")
plt.savefig("example.pdf", bbox_extra_artists=(legend,), bbox_inches='tight')
"""

