#
# Import general modules needed
#
import astropy.units as u
import numpy as np
import argparse
import yaml
import sys
# add dithering module to path and load it
sys.path.append('./py')
import dithering

parser = argparse.ArgumentParser(description="This is a script to run dithering on different sources with different dithering options")
parser.add_argument("-y", dest="yaml", help="A yaml file that has the information for running.", required=True)
parser.add_argument("-v", dest="verbose", help="verbosity of the script", default=0)
parsed_args = parser.parse_args()
yaml_file = parsed_args.yaml
yaml_input = yaml.load(open(yaml_file, "r"))

# start parsing the yaml file
sources_info = yaml_input['sources']
dithering_info = yaml_input['dithering']
telescope_info = yaml_input['telescope']

# source info
num_sources  = sources_info['num_sources']

# telescope info
# TODO: make it more generic for different configurations
blur    = True
offsets = telescope_info['offsets']
randoms = telescope_info['random_offsets']
scale   = telescope_info['scale']
config_filename = "config/desi-blur.yaml"
if offsets:
    config_filename = "config/desi-blur-offset.yaml"
if randoms:
    config_filename = "config/desi-blur-offset-stochastic.yaml"
    
# dithering info
telescope_dithering = dithering_info['telescope_dithering']
positioner_dithering = dithering_info['positioner_dithering']

dithering = dithering.dithering(config_file=config_filename)
alt_bore, az_bore = telescope_info['telescope_boresight']
dithering.set_boresight_position(alt_bore*u.deg, az_bore*u.deg)

for i in range(1, num_sources+1):
    source_type = sources_info[i]['source_type']
    if source_type == "qso" or source_type == "QSO":
        source    = dithering.generate_source(disk_fraction=0., bulge_fraction=0., half_light_disk=0., half_light_bulge=0.)
    elif source_type == "elg" or source_type == "ELG":
        half_light_radius = sources_info[i]['half_light_radius']
        source    = dithering.generate_source(disk_fraction=1., bulge_fraction=0., half_light_disk=half_light_radius, half_light_bulge=0.)
    elif source_type == "lrg" or source_type == "LRG":
        half_light_radius = sources_info[i]['half_light_radius']
        source    = dithering.generate_source(disk_fraction=0., bulge_fraction=1., half_light_disk=0., half_light_bulge=half_light_radius)
    alt, az = sources_info[i]['source_position']
    dithering.set_source_position(alt*u.deg, az*u.deg)
    dithering.set_focal_plane_position()
    dithering.run_simulation(source_type, *source, report=True)
    
"""





source    = dithering.generate_source(disk_fraction=0., bulge_fraction=0., half_light_disk=0., half_light_bulge=0.)
source_type = "qso"
dithering.set_source_position(20.*u.deg, 25.*u.deg)
dithering.set_boresight_position(20.*u.deg, 24.5*u.deg)
dithering.set_focal_plane_position()
dithering.run_simulation(source_type, *source, report=True)

source    = dithering.generate_source(disk_fraction=1., bulge_fraction=0., half_light_disk=0.5, half_light_bulge=0.)
source_type = "elg"
dithering.set_source_position(20.*u.deg, 25.*u.deg)
dithering.set_boresight_position(20.*u.deg, 24.5*u.deg)
dithering.set_focal_plane_position()
dithering.run_simulation(source_type, *source, report=True)

source    = dithering.generate_source(disk_fraction=0., bulge_fraction=1., half_light_disk=0., half_light_bulge=0.5)
source_type = "elg"
dithering.set_source_position(20.*u.deg, 25.*u.deg)
dithering.set_boresight_position(20.*u.deg, 24.5*u.deg)
dithering.set_focal_plane_position()
dithering.run_simulation(source_type, *source, report=True)
"""
