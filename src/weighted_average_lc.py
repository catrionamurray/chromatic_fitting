from .imports import *
# import sys
from .utils import *
from astropy.stats import sigma_clip
from scipy.stats import median_abs_deviation

def running_box(x,y,boxsize,operation):
    def clipped_std(x):
        return np.ma.std(sigma_clip(x,sigma=3))

    if operation == "std":
        op = np.nanstd
    elif operation == 'median':
        op = np.nanmedian
    elif operation == 'mean':
        op = np.nanmean
    elif operation == "clipped_std":
        op = clipped_std
    else:
        print("No running function selected - choose std, median or mean.")
        return np.nan

    l = len(x)
    boxsize = boxsize / 2
    dy = np.zeros((np.size(x)))

    for i in range(0, l):
        s = x[i]

        box_s = find_nearest(x, (s - boxsize))
        box_e = find_nearest(x, (s + boxsize))

        # calculate [OPERATION] for the current window
        try:
            y_ext = y[box_s:box_e + 1]
            d = op(y_ext)
            if not np.isnan(d):
                dy[i] = d

        except Exception as e:
            print(e)

    dy = np.ma.masked_where(dy == 0, dy)
    dy = np.ma.masked_invalid(dy)
    return dy


def weighted_avg_lc(time, flux, fluxe, wavelength, wavelength_range):
    #     mask bad wavelength
    boxsize= 0.5

    # find nearest index to start and end of wavelength range
    i_start = find_nearest(wavelength, wavelength_range[0])
    i_end = find_nearest(wavelength, wavelength_range[1])
    print(i_start, i_end, wavelength[i_start], wavelength[i_end])

    # Calculate 1/sigma^2 using the error in the flux
    inv_var = [np.ma.divide(1, fe ** 2) for fe in fluxe[i_start:i_end]]
    # Make a series of weights that sum to one
    weight = inv_var / np.nansum(inv_var)

    plt.plot(wavelength[i_start:i_end], np.nanmedian(fluxe[i_start:i_end], axis=1))
    plt.show()
    plt.close()

    # Time-averaged weight for each wavelength:
    sigma_sq = np.nanmedian(weight, axis=1)

    # Calculate the running median and standard dev of the weights (remove outliers!)
    running_median = running_box(wavelength[i_start:i_end], sigma_sq, boxsize* u.Unit("micron"), 'median')
    running_std = running_box(wavelength[i_start:i_end], sigma_sq, boxsize* u.Unit("micron"), 'clipped_std')

    # Remove the outlier weights
    new_sigma_sq = sigma_sq[sigma_sq < (running_median + (5 * running_std))]
    new_wav = wavelength[i_start:i_end][sigma_sq < (running_median + (5 * running_std))]
    new_flux = flux[i_start:i_end][sigma_sq < (running_median + (5 * running_std))]
    new_fluxe = fluxe[i_start:i_end][sigma_sq < (running_median + (5 * running_std))]
    new_inv_var = [np.ma.divide(1, fe** 2) for fe in new_fluxe]
    new_weight = new_inv_var / np.nansum(new_inv_var)


    plt.plot(wavelength[i_start:i_end], sigma_sq, c='orange')
    plt.plot(wavelength[i_start:i_end], running_median, c='green')
    plt.plot(new_wav, new_sigma_sq, c='b')
    plt.ylabel("Time-averaged (1/sigma)^2")
    plt.xlabel("Wavelength")
    plt.show()
    plt.close()

    sigma_sq = np.nanmedian(new_weight, axis=0)
    # Do a traditional sigma-clip of the weights
    sclip_sigma_sq = sigma_clip(sigma_sq, 3)
    plt.plot(time, sigma_sq)
    plt.plot(time, sclip_sigma_sq, c='orange')
    plt.ylabel("Wavelength-averaged (1/sigma)^2")
    plt.xlabel("Time")
    plt.show()
    plt.close()

    # weight the LC by the inverse varirance:
    weighted_flux = np.nansum([f * iv for f, iv in zip(new_flux, new_weight)], axis=0)

    norm_weight_wavelength = np.nansum(new_weight, axis=0)
    norm_weight_time_wavelength = norm_weight_wavelength / np.nansum(norm_weight_wavelength)
    weighted_lc = weighted_flux / norm_weight_time_wavelength
    weighted_lc = weighted_lc / np.nanmedian(weighted_lc)
    print(np.shape(weighted_lc))
    mad = median_abs_deviation(weighted_lc)
    print("MAD (entire LC) =", 1000000 * mad, "ppm")
    mad = median_abs_deviation(weighted_lc[-50:])
    print("MAD (Last 50 points) =", 1000000 * mad, "ppm")

    # bin data in 5 minute time bins
    bt, bf, bdf = bin_data(time, weighted_lc, mins_jd=5 / 1440.)
    bmad = median_abs_deviation(bf)
    print("MAD (5-min binned) =", 1000000 * bmad, "ppm")

    # plot weighted average LC
    plt.plot(time, weighted_lc, 'c.')
    # plt.plot(time[-50:], weighted_lc[-50:], c="orange", marker='.', linestyle="None", label="Last 50 points")
    plt.plot(bt, bf, 'k.')
    plt.errorbar(bt, bf, bdf, linestyle="None", c='k', alpha=0.3)
    plt.ylabel("Flux")
    plt.xlabel("Time (BJD_TDB)")
    # plt.title("Weighted Mean of Wavelength Flux from " + wavelength_range[0] + " to " + wavelength_range[1] + " microns")
    plt.legend()
    plt.ylim(0.965, 1.01)
    plt.tight_layout()
    plt.show()

    # norm_err = fluxe[i_start:i_end] / flux[i_start:i_end]
    weighted_err = 1.253 * np.nanstd([r for r in flux[i_start:i_end]], axis=0) / np.sqrt(
        len(wavelength[i_start:i_end]))

    return weighted_lc, weighted_err

