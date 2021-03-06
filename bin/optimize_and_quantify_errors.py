import math
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rcParams
from astropy.io import fits
from astropy import table
import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import CloughTocher2DInterpolator
import pickle
import sys
import astropy.units as u
from astropy import wcs

plt.rcParams['font.family'] = "sans-serif"       
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2

data_path = "../data/1e-12um_RMS"
radius = [100.0, 70.0, 60.0, 50.0, 45.0, 
          40.0, 35.0, 30.0, 25.0, 15.0, 
          10.0, 6.0]

def twoD_Gaussian(xy_tuple, amplitude, xo, yo, sigma_x, sigma_y):
    (x, y) = xy_tuple
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    g = amplitude * np.exp( - ( (x-xo)**2/(2*sigma_x**2) + (y-yo)**2/(2*sigma_y**2) ) )
    return g.ravel()

def oneD_Gaussian(xy_tuple, amplitude, xo, yo, sigma):
    (x, y) = xy_tuple
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    g = amplitude * np.exp( - ( (x-xo)**2/(2*sigma**2) + (y-yo)**2/(2*sigma**2) ) )
    return g.ravel()

def twoD_ellipticGaussian(xy_tuple, amplitude, xo, yo, sigma_1, sigma_2, phi):
    (x, y) = xy_tuple
    amplitude = float(amplitude)
    xo = float(xo)
    yo = float(yo)
    sigma_1 = float(sigma_1)
    sigma_2 = float(sigma_2)
    phi     = float(phi)
    A = (np.cos(phi) / sigma_1)**2 + (np.sin(phi) / sigma_2)**2
    B = (np.sin(phi) / sigma_1)**2 + (np.cos(phi) / sigma_2)**2
    C = 2 * np.sin(phi) * np.cos(phi)*(1/sigma_1**2 - 1/sigma_2**2)
    g = amplitude * np.exp( -0.5 * ( A*(x-xo)**2 + B*(y-yo)**2 + C*(x-xo)*(y-yo) ) )
    return g.ravel()


def fit_simulation(x, y, signal, max_dim=None):
    # normalize xs, ys, signal
    nominal_psf = 55.
    max_x = max(np.fabs(x))
    max_y = max(np.fabs(y))
    if max_dim == None:
        max_dim = 70.#max(max_x, max_y)
    max_s = max(signal)
    x = x/max_dim
    y = y/max_dim
    signal = signal/max_s
    tmp_sigma = nominal_psf/max_dim
    
    coordinates = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
    data = np.array(signal).ravel()
    
    start_index = np.argmax(data)
    initial_guess = (max(signal), 
                     x[start_index], y[start_index], 
                     nominal_psf/max_x, nominal_psf/max_y, math.pi/95.)
    
    signal_bounds = [max(signal)/5., max(signal)*4.]
    sigma_bounds  = [tmp_sigma-0.1 if tmp_sigma-0.1>0. else 0.01, 2.1]
    popt, pcov = opt.curve_fit(twoD_ellipticGaussian,
                               coordinates,
                               data,
                               p0=initial_guess,
                               bounds=([signal_bounds[0], min(x), min(y),  sigma_bounds[0], sigma_bounds[0], -math.pi], 
                                       [signal_bounds[1], max(x), max(y),  sigma_bounds[1], sigma_bounds[1], math.pi]) )
    #print(popt)
    return popt[1]*max_dim, popt[2]*max_dim, popt[3]*max_dim, popt[4]*max_dim, popt[5]

def fit_simulation2(x, y, signal, max_dim=None):
    # normalize xs, ys, signal
    nominal_psf = 55.
    max_s = max(signal)
        
    x = x/max_dim
    y = y/max_dim
    nominal_psf = nominal_psf/max_dim
    signal = signal/max_s

    coordinates = np.vstack((np.array(x).ravel(), np.array(y).ravel()))
    data = np.array(signal).ravel()
    
    start_index = np.argmax(data)

    initial_guess = (max(signal), x[start_index], y[start_index], nominal_psf)
    signal_bounds = [max(signal)/1.5, max(signal)*3.]
    sigma_bounds  = [20./max_dim, 100./max_dim]
    popt, pcov = opt.curve_fit(oneD_Gaussian,
                               coordinates,
                               data,
                               method='lm',
                               p0=initial_guess)
    
    return popt[1]*max_dim, popt[2]*max_dim, popt[3]*max_dim, None, None

pickled_data = pickle.load(open("offset_field.pkl", "rb"))
xs = np.array(pickled_data["x"])
ys = np.array(pickled_data["y"])
xys = list(zip(xs, ys))
interpolator_org_x = pickled_data["interpolator_x"]
interpolator_org_y = pickled_data["interpolator_y"]

def run_analysis_1pt(pattern, search_radius1):
    filename1 = "{}/{:.1f}um/result_{}_seeing0.01_blur_wlenoffset-randoffset_combined.fits".format(data_path, search_radius1, pattern)
    hdus1 = fits.open(filename1)
    
    dither_pos_xs1  = hdus1[1].data["dither_pos_x"]
    dither_pos_ys1  = hdus1[1].data["dither_pos_y"]
    calc_signals1   = hdus1[1].data["calc_signals"]
    focal_x         = hdus1[1].data["focal_x"]
    focal_y         = hdus1[1].data["focal_y"]
    mag             = hdus1[1].data["mag"]

    calc_offset_x = []
    calc_offset_y = []
    for iFiber in range(len(focal_x)):
        #if mag[iFiber, 0] >=20.75:
        #    continue
        x = dither_pos_xs1[iFiber]
        y = dither_pos_xs1[iFiber]
        signals = calc_signals1[iFiber]
        x_offset, y_offset, sigma_1, sigma_2, angle = fit_simulation2(x, y, signals, 2.5*search_radius1)
        calc_offset_x.append(x_offset)
        calc_offset_y.append(y_offset)

    calc_offset_x = np.array(calc_offset_x)
    calc_offset_y = np.array(calc_offset_y)
    calc_offset_r = np.sqrt(calc_offset_x**2 + calc_offset_y**2)

    return focal_x, focal_y, calc_offset_x, calc_offset_y

def run_analysis_2pt(pattern, search_radius1, search_radius2):
    # two files to combine
    filename1 = "{}/{:.1f}um/result_{}_seeing0.01_blur_wlenoffset-randoffset_combined.fits".format(data_path, search_radius1, pattern)
    filename2 = "{}/{:.1f}um/result_{}_seeing0.01_blur_wlenoffset-randoffset_combined.fits".format(data_path, search_radius2, pattern)
    # tables corresponding to the two files
    hdus1 = fits.open(filename1)
    hdus2 = fits.open(filename2)

    dither_pos_xs1  = hdus1[1].data["dither_pos_x"]
    dither_pos_xs2  = hdus2[1].data["dither_pos_x"]
    dither_pos_ys1  = hdus1[1].data["dither_pos_y"]
    dither_pos_ys2  = hdus2[1].data["dither_pos_y"]
    calc_signals1   = hdus1[1].data["calc_signals"]
    calc_signals2   = hdus2[1].data["calc_signals"]
    focal_x         = hdus1[1].data["focal_x"]
    focal_y         = hdus1[1].data["focal_y"]
    focal_x_sanity  = hdus2[1].data["focal_x"]
    focal_y_sanity  = hdus2[1].data["focal_y"]
    mag             = hdus1[1].data["mag"]

    calc_offset_x = []
    calc_offset_y = []
    for iFiber in range(len(focal_x)):
        #if mag[iFiber, 0] >=20.75:
        #    continue
        focal_x_indices = (focal_x_sanity==focal_x[iFiber])
        focal_y_indices = (focal_y_sanity==focal_y[iFiber])
        index = np.logical_and(focal_x_indices, focal_y_indices)
        iFiber_sanity = np.where(index)[0]
        
        if ((focal_x[iFiber]==focal_x_sanity[iFiber_sanity[0]]) and (focal_y[iFiber]==focal_y_sanity[iFiber_sanity[0]])):
            x = np.append(dither_pos_xs1[iFiber], dither_pos_xs2[iFiber_sanity[0]][1:])
            y = np.append(dither_pos_xs1[iFiber], dither_pos_ys2[iFiber_sanity[0]][1:])
            signals = np.append(calc_signals1[iFiber], calc_signals2[iFiber_sanity[0]][1:])
            x_offset, y_offset, sigma_1, sigma_2, angle = fit_simulation2(x, y, signals, 2.5*max(search_radius1, search_radius2))
            calc_offset_x.append(x_offset)
            calc_offset_y.append(y_offset)
        else:
            print("for index {}, focal coordinates did not match. skipping...".format(iFiber))
            print(focal_x[iFiber], focal_y[iFiber])
            print(focal_x_sanity[iFiber], focal_y_sanity[iFiber])
            continue

    calc_offset_x = np.array(calc_offset_x)
    calc_offset_y = np.array(calc_offset_y)
    calc_offset_r = np.sqrt(calc_offset_x**2 + calc_offset_y**2)

    return focal_x, focal_y, calc_offset_x, calc_offset_y

def run_analysis_3pt(pattern, search_radius1, search_radius2, search_radius3):
    # three files to combine
    filename1 = "{}/{:.1f}um/result_{}_seeing0.01_blur_wlenoffset-randoffset_combined.fits".format(data_path, search_radius1, pattern)
    filename2 = "{}/{:.1f}um/result_{}_seeing0.01_blur_wlenoffset-randoffset_combined.fits".format(data_path, search_radius2, pattern)
    filename3 = "{}/{:.1f}um/result_{}_seeing0.01_blur_wlenoffset-randoffset_combined.fits".format(data_path, search_radius3, pattern)
    # tables corresponding to the two files
    hdus1 = fits.open(filename1)
    hdus2 = fits.open(filename2)
    hdus3 = fits.open(filename3)

    dither_pos_xs1  = hdus1[1].data["dither_pos_x"]
    dither_pos_xs2  = hdus2[1].data["dither_pos_x"]
    dither_pos_xs3  = hdus3[1].data["dither_pos_x"]
    dither_pos_ys1  = hdus1[1].data["dither_pos_y"]
    dither_pos_ys2  = hdus2[1].data["dither_pos_y"]
    dither_pos_ys3  = hdus3[1].data["dither_pos_y"]
    calc_signals1   = hdus1[1].data["calc_signals"]
    calc_signals2   = hdus2[1].data["calc_signals"]
    calc_signals3   = hdus3[1].data["calc_signals"]
    focal_x         = hdus1[1].data["focal_x"]
    focal_y         = hdus1[1].data["focal_y"]
    focal_x_sanity1 = hdus2[1].data["focal_x"]
    focal_y_sanity1 = hdus2[1].data["focal_y"]
    focal_x_sanity2 = hdus3[1].data["focal_x"]
    focal_y_sanity2 = hdus3[1].data["focal_y"]
        
    calc_offset_x = []
    calc_offset_y = []
    for iFiber in range(len(focal_x)):
    
        focal_x_indices1 = (focal_x_sanity1==focal_x[iFiber])
        focal_y_indices1 = (focal_y_sanity1==focal_y[iFiber])
        index1 = np.logical_and(focal_x_indices1, focal_y_indices1)
        iFiber1 = np.where(index1)[0]

        focal_x_indices2 = (focal_x_sanity2==focal_x[iFiber])
        focal_y_indices2 = (focal_y_sanity2==focal_y[iFiber])
        index2 = np.logical_and(focal_x_indices2, focal_y_indices2)
        iFiber2 = np.where(index2)[0]
        
        x = np.append(np.append(dither_pos_xs1[iFiber], dither_pos_xs2[iFiber1[0]][1:]), dither_pos_xs3[iFiber2[0]][1:])
        y = np.append(np.append(dither_pos_xs1[iFiber], dither_pos_ys2[iFiber1[0]][1:]), dither_pos_ys3[iFiber2[0]][1:])
        signals = np.append(np.append(calc_signals1[iFiber], calc_signals2[iFiber2[0]][1:]), calc_signals3[iFiber2[0]][1:])
        
        x_offset, y_offset, sigma_1, sigma_2, angle = fit_simulation2(x, y, signals, 2*max(search_radius1, search_radius2, search_radius3))
        calc_offset_x.append(x_offset)
        calc_offset_y.append(y_offset)

    calc_offset_x = np.array(calc_offset_x)
    calc_offset_y = np.array(calc_offset_y)
    calc_offset_r = np.sqrt(calc_offset_x**2 + calc_offset_y**2)

    return focal_x, focal_y, calc_offset_x, calc_offset_y

def run_1pt(search_radius1):
    fig = plt.figure(figsize=(25, 25))
    max_value = 35.#max(drs)
    
    # calculate the values for a combination of search radii
    focal_x, focal_y, calc_offset_x, calc_offset_y = run_analysis_1pt("triangle", 
                                                                      search_radius1)
    # put the numbers into interpolators
    calc_offset_r = np.sqrt(calc_offset_x**2 + calc_offset_y**2)
    selected      = calc_offset_r <= max_value
    focal_xy      = list(zip(focal_x[selected], focal_y[selected]))
    interpolator_x = CloughTocher2DInterpolator(focal_xy, calc_offset_x[selected])
    interpolator_y = CloughTocher2DInterpolator(focal_xy, calc_offset_y[selected])

    # calculate the interpolated values at the grid points
    cx = interpolator_x(xys)
    cy = interpolator_y(xys)
    cr = np.sqrt(cx**2+cy**2)
    # draw the figure for calculated offsets less than 50um
    r = 410

    # calculate the interpolated values at the grid points
    cx_org = interpolator_org_x(xys)
    cy_org = interpolator_org_y(xys)
    cr_org = np.sqrt(cx_org**2+cy_org**2)

    # draw the figure for calculated offsets less than 50um    
    diff_cx = cx_org-cx
    diff_cy = cy_org-cy
    diff_cr = np.sqrt(diff_cx**2 + diff_cy**2)

    sampling = 2
    sampling_check = np.full(len(cr), False)
    sampling_check[::sampling] = True
    nan_check_calc = cr==cr
    nan_check_in   = cr_org==cr_org
    nan_check_res  = diff_cr==diff_cr
    max_val_check  = cr<=max_value
    last_selection = np.logical_and(sampling_check,
                                    (np.logical_and( np.logical_and(nan_check_calc, nan_check_in),
                                                     np.logical_and(nan_check_res, max_val_check))))
    
    plt.subplot(221)
    plt.quiver(xs[last_selection], ys[last_selection],
               np.array(cx_org[last_selection]), np.array(cy_org[last_selection]),
               cr_org[last_selection])
    plt.gca().set_aspect('equal','datalim')
    plt.gca().add_artist(plt.Circle((0,0), r, color = 'r', fill=False))
    cbar = plt.colorbar(fraction=0.04, pad=0.04)
    cbar.set_label("Offsets [um]")
    plt.axis('off')
    plt.title("Input Offsets")
    plt.tight_layout()

    plt.subplot(222)
    plt.quiver(xs[last_selection], ys[last_selection],
               -np.array(cx[last_selection]), -np.array(cy[last_selection]),
               cr[last_selection])
    plt.gca().set_aspect('equal','datalim')
    plt.gca().add_artist(plt.Circle((0,0), r, color = 'r', fill=False))
    cbar = plt.colorbar(fraction=0.04, pad=0.04)
    cbar.set_label("Calc Offsets [um]")
    plt.axis('off')
    plt.title("Calculated Offsets")
    plt.tight_layout()
    plt.title("{:.1f}um Triangulation Pattern".format(search_radius1))

    plt.subplot(223)
    plt.quiver(xs[last_selection], ys[last_selection],
               np.array(diff_cx[last_selection]), np.array(diff_cy[last_selection]),
               diff_cr[last_selection])
    plt.gca().set_aspect('equal','datalim')
    plt.gca().add_artist(plt.Circle((0,0), r, color = 'r', fill=False))
    cbar = plt.colorbar(fraction=0.04, pad=0.04)
    cbar.set_label("Residual Offsets [um]")
    plt.axis('off')
    plt.title("Residuals")
    plt.tight_layout()

    plt.subplot(224)
    plt.hist(cr_org[last_selection], bins=100, label="known offsets", alpha=.5, density=True)
    plt.hist(diff_cr[last_selection], bins=100, label="Residuals", alpha=.5, density=True)
    plt.legend()

    plt.savefig("results_for_{}_only.png".format(search_radius1))
    plt.savefig("results_for_{}_only.pdf".format(search_radius1))
    
    plt.close(fig)
    
    results = {}
    results["calc_offset_x"] = cx
    results["calc_offset_y"] = cy
    results["calc_offset_r"] = cr
    results["known_offset_x"] = cx_org
    results["known_offset_y"] = cy_org
    results["known_offset_r"] = cr_org
    results["residual_x"] = diff_cx
    results["residual_y"] = diff_cy
    results["residual_r"] = diff_cr
    pickle.dump(results, open("results_{}_only.pkl".format(search_radius1), "wb"))

def run_2pt(search_radius1, search_radius2, search_radius3=None):
    max_value = 50.
    
    # calculate the values for a combination of search radii
    if search_radius3 is None:
        focal_x, focal_y, calc_offset_x, calc_offset_y = run_analysis_2pt("triangle", 
                                                                          search_radius1, search_radius2)
    else:
        focal_x, focal_y, calc_offset_x, calc_offset_y = run_analysis_3pt("triangle", 
                                                                          search_radius1, search_radius2, search_radius3)
    # put the numbers into interpolators
    calc_offset_r = np.sqrt(calc_offset_x**2 + calc_offset_y**2)
    selected      = calc_offset_r <= max_value
    focal_xy      = list(zip(focal_x[selected], focal_y[selected]))
    interpolator_x = CloughTocher2DInterpolator(focal_xy, calc_offset_x[selected])
    interpolator_y = CloughTocher2DInterpolator(focal_xy, calc_offset_y[selected])

    # calculate the interpolated values at the grid points
    cx = interpolator_x(xys)
    cy = interpolator_y(xys)
    cr = np.sqrt(cx**2+cy**2)
    # draw the figure for calculated offsets less than 50um
    r = 410

    # calculate the interpolated values at the grid points
    cx_org = interpolator_org_x(xys)
    cy_org = interpolator_org_y(xys)
    cr_org = np.sqrt(cx_org**2+cy_org**2)

    # draw the figure for calculated offsets less than 50um
    diff_cx = cx_org-cx
    diff_cy = cy_org-cy
    selected3 = (diff_cx == diff_cx)
    diff_cr = np.sqrt(diff_cx**2 + diff_cy**2)

    sampling = 2
    sampling_check = np.full(len(cr), False)
    sampling_check[::sampling] = True
    nan_check_calc = cr==cr
    nan_check_in   = cr_org==cr_org
    nan_check_res  = diff_cr==diff_cr
    max_val_check  = cr<=max_value
    last_selection = np.logical_and(sampling_check,
                                    (np.logical_and(np.logical_and(nan_check_calc, nan_check_in),
                                                    np.logical_and(nan_check_res, max_val_check))))
    r = 410

    fig = plt.figure(figsize=(25, 25))
    
    plt.subplot(221)
    plt.quiver(xs[last_selection], ys[last_selection],
               np.array(cx_org[last_selection]), np.array(cy_org[last_selection]),
               cr_org[last_selection])
    plt.gca().set_aspect('equal','datalim')
    plt.gca().add_artist(plt.Circle((0,0), r, color = 'r', fill=False))
    cbar = plt.colorbar(fraction=0.04, pad=0.04)
    cbar.set_label("Offsets [um]")
    plt.axis('off')
    plt.title("Input Offsets")
    plt.tight_layout()

    plt.subplot(222)
    plt.quiver(xs[last_selection], ys[last_selection],
               -np.array(cx[last_selection]), -np.array(cy[last_selection]),
               cr[last_selection])
    plt.gca().set_aspect('equal','datalim')
    plt.gca().add_artist(plt.Circle((0,0), r, color = 'r', fill=False))
    cbar = plt.colorbar(fraction=0.04, pad=0.04)
    cbar.set_label("Calc Offsets [um]")
    plt.axis('off')
    plt.title("Calculated Offsets")
    plt.tight_layout()
    if search_radius3 is None:
        plt.title("{:.1f}um + {:.1f}um Triangulation Pattern".format(search_radius1, search_radius2))
    else:
        plt.title("{:.1f}um + {:.1f}um +{:.1f}um Triangulation Pattern".format(search_radius1, search_radius2, search_radius3))

    plt.subplot(223)
    plt.quiver(xs[last_selection], ys[last_selection], np.array(diff_cx[last_selection]), np.array(diff_cy[last_selection]), diff_cr[last_selection])
    plt.gca().set_aspect('equal','datalim')
    plt.gca().add_artist(plt.Circle((0,0), r, color = 'r', fill=False))
    cbar = plt.colorbar(fraction=0.04, pad=0.04)
    cbar.set_label("Residual Offsets [um]")
    plt.axis('off')
    plt.title("Residuals")
    plt.tight_layout()

    plt.subplot(224)
    plt.hist(cr_org[last_selection], bins=100, label="known offsets", alpha=.5,
             density=True, histtype='step', color='red', linewidth=3)
    plt.hist(diff_cr[last_selection], bins=100, label="Residuals", alpha=.5,
             density=True, histtype='step', color='blue', linewidth=3)
    plt.legend()
    plt.xlabel("Offset [um]")
    if search_radius3 is None:
        plt.savefig("results_for_{}_{}.png".format(search_radius1, search_radius2))
        plt.savefig("results_for_{}_{}.pdf".format(search_radius1, search_radius2))
    else:
        plt.savefig("results_for_{}_{}_{}.png".format(search_radius1, search_radius2, search_radius3))
        plt.savefig("results_for_{}_{}_{}.pdf".format(search_radius1, search_radius2, search_radius3))
        
    plt.close(fig)

    fig = plt.figure(figsize=(6,6))
    plt.hist(cr_org[last_selection], bins=100, label="known offsets", alpha=.5,
             density=True, cumulative=True, histtype='step', color='red', linewidth=3)
    plt.hist(diff_cr[last_selection], bins=100, label="Residuals", alpha=.5,
             density=True, cumulative=True, histtype='step', color='blue', linewidth=3)
    plt.legend(loc=2)
    plt.xlabel("Offset [um]")
    if search_radius3 is None:
        plt.savefig("results_for_{}_{}_cumulative.png".format(search_radius1, search_radius2))
        plt.savefig("results_for_{}_{}_cumulative.pdf".format(search_radius1, search_radius2))
    else:
        plt.savefig("results_for_{}_{}_{}_cumulative.png".format(search_radius1, search_radius2, search_radius3))
        plt.savefig("results_for_{}_{}_{}_cumulative.pdf".format(search_radius1, search_radius2, search_radius3))
    plt.close(fig)
    
    results = {}
    results["calc_offset_x"] = cx
    results["calc_offset_y"] = cy
    results["calc_offset_r"] = cr
    results["known_offset_x"] = cx_org
    results["known_offset_y"] = cy_org
    results["known_offset_r"] = cr_org
    results["residual_x"] = diff_cx
    results["residual_y"] = diff_cy
    results["residual_r"] = diff_cr
    if search_radius3 is None:
        pickle.dump(results, open("results_{}_{}.pkl".format(search_radius1, search_radius2), "wb"))
    else:
        pickle.dump(results, open("results_{}_{}_{}.pkl".format(search_radius1, search_radius2, search_radius3), "wb"))

"""
for r1 in radius:
    for r2 in radius:
        if r2 >= r1:
            continue
        try:
            run_2pt(r1, r2)
        except:
            continue

for r1 in radius:
    try:
        run_1pt(r1)
    except:
        continue
"""
    
"""
run_1pt(35.0)
run_1pt(70.0)
run_1pt(50.0)
run_2pt(70.0, 35.0)
run_2pt(70.0, 50.0)
run_2pt(50.0, 35.0)
"""
#run_1pt(35.0)
run_2pt(50.0, 35.0)
#run_2pt(100.0, 50.0, search_radius3=10.0)
"""
for r1 in radius:
    for r2 in radius:
        if r2>=r1:
            continue
        else:
            for r3 in radius:
                if r3>=r2:
                    continue
                else:
                    try:
                        run_2pt(r1, r2, search_radius3=r3)
                    except:
                        print("problem with {} - {} - {} combination".format(r1, r2 ,r3))


"""
