#
# Import general modules needed
#
import astropy.units as u
import numpy as np
import argparse
import yaml
import random
import sys
import matplotlib.pyplot as plt
# add dithering module to path and load it
sys.path.append('./py')
import dithering

parser = argparse.ArgumentParser(description="This is a script to run dithering on different sources with different dithering options")
parser.add_argument("-y", dest="yaml", help="A yaml file that has the information for running.", required=True)
parser.add_argument("-o", dest="output", help="An output file foe the outputs")
parser.add_argument("-v", dest="verbose", help="verbosity of the script", default=0)
parsed_args = parser.parse_args()
yaml_file = parsed_args.yaml
output_file = parsed_args.output

# start parsing the yaml file
yaml_input = yaml.load(open(yaml_file, "r"))
sources_info = yaml_input['sources']
dithering_info = yaml_input['dithering']
telescope_info = yaml_input['telescope']

# source info
num_sources  = sources_info['num_sources']

# telescope info
config_filename = telescope_info['telescope_config']
    
# dithering info
telescope_dithering = dithering_info['telescope_dithering']
positioner_dithering = dithering_info['positioner_dithering']

# create the object
dithering = dithering.dithering(config_file=config_filename, output_file=output_file)
alt_bore, az_bore = telescope_info['telescope_boresight']
dithering.set_boresight_position(alt_bore*u.deg, az_bore*u.deg)

# generate a set of random values to be used
num_randoms = yaml_input['num_randoms']#1000
x_offsets   = np.zeros(num_randoms)
y_offsets   = np.zeros(num_randoms)
az_offsets  = np.zeros(num_randoms)
alt_offsets = np.zeros(num_randoms)
for i in range(num_randoms):
    x_offsets[i] = random.gauss(0, 1)*dithering.platescale.value
    y_offsets[i] = random.gauss(0, 1)*dithering.platescale.value
    az_offsets[i]  = random.gauss(0, 1)
    alt_offsets[i] = random.gauss(0, 1)

sources = []
for i in range(1, num_sources+1):
    results_b_x = []; results_r_x = []; results_z_x = []
    results_b_y = []; results_r_y = []; results_z_y = []
    results_b_xy = []; results_r_xy = []; results_z_xy = []
    results_b_alt = []; results_r_alt = []; results_z_alt = []
    results_b_az = []; results_r_az = []; results_z_az = []
    results_b_bore = []; results_r_bore = []; results_z_bore = []
    source_type = sources_info[i]['source_type']
    if source_type == "qso" or source_type == "QSO":
        source    = dithering.generate_source(disk_fraction=0., bulge_fraction=0., half_light_disk=0., half_light_bulge=0.)
        half_light_radius = 0
    elif source_type == "elg" or source_type == "ELG":
        half_light_radius = sources_info[i]['half_light_radius']
        source    = dithering.generate_source(disk_fraction=1., bulge_fraction=0., half_light_disk=half_light_radius, half_light_bulge=0.)
    elif source_type == "lrg" or source_type == "LRG":
        half_light_radius = sources_info[i]['half_light_radius']
        source    = dithering.generate_source(disk_fraction=0., bulge_fraction=1., half_light_disk=0., half_light_bulge=half_light_radius)
    alt, az = sources_info[i]['source_position']
    theta_0, phi_0 = sources_info[i]['positioner_initials']
    dithering.set_source_position(alt*u.deg, az*u.deg)
    dithering.set_focal_plane_position()
    dithering.set_theta_0(theta_0*u.deg)
    dithering.set_phi_0(phi_0*u.deg)
    for j in range(num_randoms):
        # only random x changes
        dithering.place_fiber([x_offsets[j], 0])
        dithering.run_simulation(source_type, *source, report=False)
        results_b_x.append(np.median(dithering.SNR['b'][0]))
        results_r_x.append(np.median(dithering.SNR['r'][0]))
        results_z_x.append(np.median(dithering.SNR['z'][0]))
        # only random y changes
        dithering.place_fiber([0, y_offsets[j]])
        dithering.run_simulation(source_type, *source, report=False)
        results_b_y.append(np.median(dithering.SNR['b'][0]))
        results_r_y.append(np.median(dithering.SNR['r'][0]))
        results_z_y.append(np.median(dithering.SNR['z'][0]))
        # both x and y changes at the same time
        dithering.place_fiber([x_offsets[j], y_offsets[j]])
        dithering.run_simulation(source_type, *source, report=False)
        results_b_xy.append(np.median(dithering.SNR['b'][0]))
        results_r_xy.append(np.median(dithering.SNR['r'][0]))
        results_z_xy.append(np.median(dithering.SNR['z'][0]))
        # only telescope boresight azimuth changes
        dithering.change_alt_az_bore_position(alt_bore*u.deg, az_bore*u.deg+az_offsets[j]*u.arcsec)
        dithering.run_simulation(source_type, *source, report=False)
        results_b_az.append(np.median(dithering.SNR['b'][0]))
        results_r_az.append(np.median(dithering.SNR['r'][0]))
        results_z_az.append(np.median(dithering.SNR['z'][0]))
        # only telescope boresight altitude changes
        dithering.change_alt_az_bore_position(alt_bore*u.deg+alt_offsets[j]*u.arcsec, az_bore*u.deg)
        dithering.run_simulation(source_type, *source, report=False)
        results_b_alt.append(np.median(dithering.SNR['b'][0]))
        results_r_alt.append(np.median(dithering.SNR['r'][0]))
        results_z_alt.append(np.median(dithering.SNR['z'][0]))
        # both  altitude and azimuth changes at the same time
        dithering.change_alt_az_bore_position(alt_bore*u.deg+alt_offsets[j]*u.arcsec, az_bore*u.deg+az_offsets[j]*u.arcsec)
        dithering.run_simulation(source_type, *source, report=False)
        results_b_bore.append(np.median(dithering.SNR['b'][0]))
        results_r_bore.append(np.median(dithering.SNR['r'][0]))
        results_z_bore.append(np.median(dithering.SNR['z'][0]))
        # move the telescope to original position - reset
        dithering.change_alt_az_bore_position(alt_bore*u.deg, az_bore*u.deg)
        
    source = {}
    source['source_type'] = source_type
    source['TAltitude']    = alt_bore
    source['TAzimuth']     = az_bore
    source['theta_0']     = theta_0
    source['phi_0']       = phi_0
    source['half_light_r']= half_light_radius
    source['focal_x']     = dithering.focal_x
    source['focal_y']     = dithering.focal_y
    source['SAltitude']   = alt
    source['SAzimuth']    = az
    source['x_offsets']   = {'x_offsets':x_offsets.flatten(), 'results_b':results_b_x,
                             'results_r':results_r_x, 'results_z':results_z_x}
    source['y_offsets']   = {'y_offsets':y_offsets.flatten(), 'results_b':results_b_y,
                             'results_r':results_r_y, 'results_z':results_z_y}
    source['xy_offsets']  = {'x_offsets':x_offsets.flatten(), 'y_offsets':y_offsets.flatten(),
                             'results_b':results_b_xy, 'results_r':results_r_xy,
                             'results_z':results_z_xy}
    source['az_offsets']  = {'az_offsets':az_offsets.flatten(), 'results_b':results_b_az,
                             'results_r':results_r_az, 'results_z':results_z_az}
    source['alt_offsets'] = {'alt_offsets':alt_offsets.flatten(), 'results_b':results_b_alt,
                             'results_r':results_r_alt, 'results_z':results_z_alt}
    source['bore_offsets']= {'az_offset':az_offsets.flatten(), 'alt_offsets':alt_offsets.flatten(),
                             'results_b':results_b_bore, 'results_r':results_r_bore,
                             'results_z':results_z_bore}
    sources.append(source)
np.savez("results.npz", sources = sources)

