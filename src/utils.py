from .imports import *

def rainbow_to_vector(r):
    # r: rainbow object
    rflux = r.fluxlike['flux']         # flux (MJy/sr)        : [n_wavelengths x n_integrations]
    rfluxe = r.fluxlike['uncertainty'] # flux error (MJy/sr)  : [n_wavelengths x n_integrations]
    rtime = r.timelike['time']/24.     # time (BJD_TDB, days) : [n_integrations]
    rwavel = r.wavelike['wavelength']  # wavelength (microns) : [n_wavelengths]
    return rflux,rfluxe,rtime,rwavel


def bin_data(jd, y, mins_jd):
    t = np.array(jd)

    split = []
    sorted_t = t
    sorted_y = y
    start = sorted_t[0]
    nextbin = sorted_t[np.absolute(sorted_t - start) > mins_jd]

    while nextbin != []:
        start = start + mins_jd
        ind_st = np.argmax(sorted_t > start)
        if len(split) > 0:
            if ind_st != split[-1]:
                split.append(ind_st)
                time = sorted_t[ind_st:]
        else:
            split.append(ind_st)
            time = sorted_t[ind_st:]
        nextbin = time[np.absolute(time - start) > mins_jd]

    times = np.split(sorted_t, split)
    ys = np.split(sorted_y, split)

    bins = np.zeros(len(times))
    binned_y = np.zeros(len(times))
    binned_err = np.zeros(len(times))

    for i in range(len(times)):
        if len(ys[i]) > 0:
            try:
                bins[i] = np.nanmean(times[i])
                binned_y[i] = np.nanmean(ys[i])
                n = len(times[i])
                # standard error in the median:
                # binned_err[i] = 1.253 * np.nanstd(ys[i]) / np.sqrt(n)
                binned_err[i] = np.nanstd(ys[i]) / np.sqrt(n)
            except Exception as e:
                pass

    bin_t = bins[binned_y != 0]
    bin_e = binned_err[binned_y != 0]
    bin_y = binned_y[binned_y != 0]

    return bin_t, bin_y, bin_e

def find_nearest(array, value):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx