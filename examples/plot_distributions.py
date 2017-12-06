import numpy as np
from  astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
# general plotting options
rcParams.update({'figure.autolayout': True})
my_cmap = plt.cm.jet
my_cmap.set_under('white',.01)
my_cmap.set_over('white', 300)
my_cmap.set_bad('white')

search_radia = [30, 50, 70, 90]

def plot(band, search_radius):

    hdulist = fits.open("../data/{}um/results.fits".format(search_radius))
    x = hdulist[1].data['known_snr_{}'.format(band)]
    y = hdulist[1].data['calc_snr_{}'.format(band)]

    for i in range(1, 10):
        hdulist = fits.open("../data/{}um/results_{}.fits".format(search_radius, i))
        x = np.append(x, hdulist[1].data['known_snr_{}'.format(band)])
        y = np.append(y, hdulist[1].data['calc_snr_{}'.format(band)])

    plt.clf()
    plt.hist2d(x, y, bins=50, cmap=my_cmap, vmin=1, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.grid(True)
    plt.xlabel("True SNR-{}".format(band))
    plt.ylabel("Calculated SNR-{}".format(band))
    plt.savefig("../figures/disto_{}um_{}.pdf".format(search_radius, band))
    plt.savefig("../figures/disto_{}um_{}.png".format(search_radius, band))

for r in search_radia:
    plot("b", float(r))
    plot("r", float(r))
    plot("z", float(r))
