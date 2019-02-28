import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl

# general plotting options
rcParams.update({'figure.autolayout': True})
my_cmap = plt.cm.jet
my_cmap.set_under('white',.01)
my_cmap.set_over('white', 300)
my_cmap.set_bad('white')

def plot(band, search_radius, rms):

    hdulist = fits.open("../data/{}.0um_RMS/{}um/results.fits".format(int(rms), search_radius))
    x = hdulist[1].data['known_snr_{}'.format(band)]
    y = hdulist[1].data['calc_snr_{}'.format(band)]

    for i in range(1, 10):
        try:
            filename = "../data/{}.0um_RMS/{}um/results_{}.fits".format(int(rms), search_radius, i)
            print(filename)
            hdulist = fits.open("../data/{}.0um_RMS/{}um/results_{}.fits".format(int(rms), search_radius, i))
            print("here")
            x = np.append(x, hdulist[1].data['known_snr_{}'.format(band)])
            y = np.append(y, hdulist[1].data['calc_snr_{}'.format(band)])
            print("there")
        except:
            continue

    print(len(x))
    plt.plot(x, y, 'o')
    plt.show()
        
    counts, xedges, yedges, img = plt.hist2d(x, y, bins=100, cmap=my_cmap, vmin=1, norm=mpl.colors.LogNorm())
    plt.show()
    x_binsize = (xedges[1]-xedges[0])
    y_binsize = (yedges[1]-yedges[0])
    num_xedges = len(xedges)
    num_yedges = len(yedges)
    errors = []
    weights = []
    for i in range(num_xedges-1):
        known_snr = xedges[i]#-x_binsize/2.
        for j in range(num_yedges-1):
            errors.append(abs(xedges[i]-yedges[j])/xedges[i])
            weights.append(counts[i][j])
    plt.clf()
    plt.hist(errors, weights=weights, bins=num_xedges)
    plt.xlim(0, 1.1)
    plt.xlabel("(SNR$_{known}$-SNR$_{calc}$)/SNR$_{known}$")
    plt.title("SNR {} results".format(band))
    plt.savefig("../figures/percent_errors_1Dhist_{}umRMS_{}um_{}.eps".format(int(rms), search_radius, band))
    plt.savefig("../figures/percent_errors_1Dhist_{}umRMS_{}um_{}.png".format(int(rms), search_radius, band))
    plt.savefig("../figures/percent_errors_1Dhist_{}umRMS_{}um_{}.pdf".format(int(rms), search_radius, band))
    plt.show()
    
    x = []
    s = []
    f = []
    for i in range(num_xedges-1):
        success = 0
        failure = 0
        for j in range(num_yedges-1):
            error = abs(xedges[i]-yedges[j])/xedges[i]
            if error > 0.15:
                failure = failure + counts[i][j]
            else:
                success = success + counts[i][j]
        if success == success:
            x.append(xedges[i])
            s.append(success/(success+failure))
            f.append(failure/(success+failure))
    plt.clf()
    plt.plot(x, f, 'x')
    plt.title("SNR {} results".format(band))
    plt.xlabel("SNR$_{known}")
    plt.ylabel("Failure Rate (error>10%)")
    plt.savefig("../figures/failure_rate_{}umRMS_{}um_{}.eps".format(int(rms), search_radius, band))
    plt.savefig("../figures/failure_rate_{}umRMS_{}um_{}.pdf".format(int(rms), search_radius, band))
    plt.savefig("../figures/failure_rate_{}umRMS_{}um_{}.png".format(int(rms), search_radius, band))
    plt.show()
    plt.clf()
    plt.plot(x, s, 'x')
    plt.title("SNR {} results".format(band))
    plt.xlabel("SNR$_{known}$")
    plt.ylabel("Success Rate (error<10%)")
    plt.savefig("../figures/success_rate_{}umRMS_{}um_{}.eps".format(int(rms), search_radius, band))
    plt.savefig("../figures/success_rate_{}umRMS_{}um_{}.pdf".format(int(rms), search_radius, band))
    plt.savefig("../figures/success_rate_{}umRMS_{}um_{}.png".format(int(rms), search_radius, band))
    plt.show()

search_radia = [35]#[30, 50, 70, 90]
rmss = [70]#[50, 70]

for rms in rmss:
    for r in search_radia:
        #plot("b", float(r), rms)
        plot("r", float(r), rms)
        #plot("z", float(r), rms)
