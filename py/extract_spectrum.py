import os
import fitsio
import random
import numpy as np
import healpy as hp
from glob import glob
from collections import defaultdict
from desitarget import desi_mask
import matplotlib.pyplot as plt

os.environ["DESI_SPECTRO_REDUX"] = "/home/tyapici/data/DESI/spectro/redux/"
os.environ["SPECPROD"] = "dc17a2"

basedir = os.path.join(os.getenv("DESI_SPECTRO_REDUX"),os.getenv("SPECPROD"),"spectra-64")
subdir = os.listdir(basedir)
basedir = os.path.join(basedir,subdir[0])
subdir = os.listdir(basedir)
basedir = os.path.join(basedir,subdir[0])
subdir = os.listdir(basedir)

specfilename = glob(os.path.join(basedir,'spectra*fits'))[0]
fm = fitsio.read(specfilename,1)
specfilename = glob(os.path.join(basedir,'spectra*fits'))[0]

bwave = fitsio.read(specfilename,"B_WAVELENGTH")
rwave = fitsio.read(specfilename,"R_WAVELENGTH")
zwave = fitsio.read(specfilename,"Z_WAVELENGTH")
wave = np.hstack([bwave,rwave,zwave])

bflux = fitsio.read(specfilename,"B_FLUX")
rflux = fitsio.read(specfilename,"R_FLUX")
zflux = fitsio.read(specfilename,"Z_FLUX")
flux = np.hstack([bflux,rflux,zflux])

def get_spectrum(source_type, idx, output=False):
    stds = np.where(fm["DESI_TARGET"] & desi_mask[source_type])[0]
    #plt.plot(fm["RA_TARGET"],fm["DEC_TARGET"],'b,')
    #plt.plot(fm["RA_TARGET"][stds],fm["DEC_TARGET"][stds],'kx')
    #plt.show()
    from scipy.interpolate import interp1d
    num_objects = len(stds)
    assert idx < num_objects
    extrapolator = interp1d(wave, flux[stds[idx]], fill_value='extrapolate')
    wavelengths = np.arange(3500., 10000.0, 0.1)
    fluxvalues = np.zeros(len(wavelengths))
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
    return wavelengths, fluxvalues

def get_random_spectrum(source_type, output=False):
    stds = np.where(fm["DESI_TARGET"] & desi_mask[source_type])[0]
    num_objects = len(stds)
    idx = random.randint(0, num_objects-1)
    return get_spectrum(source_type, idx, output)

if __name__=="__main__":
    print(get_spectrum("STD_FSTAR", 0))
