import os
import fitsio
import random
import numpy as np
import healpy as hp
from glob import glob
from collections import defaultdict
from desitarget.targetmask import desi_mask
import matplotlib.pyplot as plt

os.environ["STAR_SPECTRA"] = "spectro/datachallenge/dc17b/spectro/redux/dc17b/spectra-64"

basedir = os.path.join(os.getenv("DESI_ROOT"),os.getenv("STAR_SPECTRA"))
specfilenames = glob(basedir+"/*/*/spectra*")

def get_spectrum(file_idx=0, source_idx=0, output=False):
    specfilename = specfilenames[file_idx]
    bwave = fitsio.read(specfilename,"B_WAVELENGTH")
    rwave = fitsio.read(specfilename,"R_WAVELENGTH")
    zwave = fitsio.read(specfilename,"Z_WAVELENGTH")
    wave = np.hstack([bwave,rwave,zwave])
    bflux = fitsio.read(specfilename,"B_FLUX")[source_idx]
    rflux = fitsio.read(specfilename,"R_FLUX")[source_idx]
    zflux = fitsio.read(specfilename,"Z_FLUX")[source_idx]
    flux = np.hstack([bflux,rflux,zflux])
    mag  = fitsio.read(specfilename, 1)["MAG"][source_idx]
    
    from scipy.interpolate import interp1d
    extrapolator = interp1d(wave, flux, fill_value='extrapolate')
    wavelengths  = np.arange(3500., 10000.0, 0.1)
    fluxvalues   = np.zeros(len(wavelengths))
    if output:
        fd = open("{}_spectrum_{}.dat".format(source_type, i), "w")
        fd.write("#   WAVELENGTH        FLUX\n#------------- -----------\n")
    for i in range(len(wavelengths)):
        wavelength = wavelengths[i]
        fluxvalue = extrapolator(wavelength)
        if fluxvalue < 0:
            fluxvalue = 0.
        fluxvalues[i] = fluxvalue
        if output:
            fd.write("     {0:.3f}     {1:.4f}\n".format(wavelengths[i], fluxvalues[i]))
    if output:
        fd.close()
    return wavelengths, fluxvalues, np.array(mag)

def get_random_spectrum(source_type="STD_BRIGHT", output=False):
    num_objs = 0
    while num_objs<=0:
        file_idx = random.randint(0, len(specfilenames)-1)
        fm       = fitsio.read(specfilenames[file_idx],1)
        stds     = np.where(fm["DESI_TARGET"] & desi_mask[source_type])[0]
        num_objs = len(stds)
    random_obj = random.randint(0, num_objs-1)
    source_idx  = stds[random_obj]
    return get_spectrum(file_idx, source_idx, output)

if __name__=="__main__":
    print(get_random_spectrum("STD_BRIGHT", 0))
