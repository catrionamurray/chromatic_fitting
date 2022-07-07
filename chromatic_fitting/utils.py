from .imports import *


def rainbow_to_vector(r, timeformat='h'):
    """ Convert Rainbow object to np.arrays
    Parameters
    ----------
        r : Rainbow object
            chromatic Rainbow object to convert into array format
        timeformat : str
            (optional, default='hours')
            The time format to use (seconds, minutes, hours, days etc.)
    Returns
    ----------
        rflux : np.array
            flux (MJy/sr)        [n_wavelengths x n_integrations]
        rfluxe : np.array
            flux error (MJy/sr)  [n_wavelengths x n_integrations]
        rtime : np.array
            time (BJD_TDB, houra) [n_integrations]
        rwavel : np.array
            wavelength (microns) [n_wavelengths]
    """
    secondformat = ['second', 'seconds', 'sec', 's']
    minuteformat = ['minute', 'minutes', 'min', 'm']
    hourformat = ['hour', 'hours', 'h']
    dayformat = ['day', 'days', 'd']
    yearformat = ['year', 'years', 'y']

    rflux = r.fluxlike['flux']  # flux (MJy/sr)        : [n_wavelengths x n_integrations]
    rfluxe = r.fluxlike['uncertainty']  # flux error (MJy/sr)  : [n_wavelengths x n_integrations]
    rtime = r.timelike['time']  # time (BJD_TDB, hours) : [n_integrations]
    rwavel = r.wavelike['wavelength']  # wavelength (microns) : [n_wavelengths]

    # change the time array into the requested format (e.g. seconds, minutes, days etc.)
    if timeformat in secondformat:
        rtime = rtime * 3600
    elif timeformat in minuteformat:
        rtime = rtime * 60
    elif timeformat in hourformat:
        # hours is the default time setting
        pass
    elif timeformat in dayformat:
        rtime = rtime / 24.
    elif timeformat in yearformat:
        rtime = rtime / (24 * 365.)
    else:
        warnings.warn("Unrecognised Time Format!")
        return

    return rflux, rfluxe, rtime, rwavel


def rainbow_to_df(r, timeformat='h'):
    """ Convert Rainbow object to pandas dataframe
    Parameters
    ----------
        r : Rainbow object
            chromatic Rainbow object to convert into pandas df format
        timeformat : str
            (optional, default='hours')
            The time format to use (seconds, minutes, hours, days etc.)
    Returns
    ----------
        pd.DataFrame
    """
    rflux, rfluxe, rtime, rwavel = rainbow_to_vector(r, timeformat)
    x, y = np.meshgrid(rtime.to_value(), rwavel.to_value())
    rainbow_dict = {f"Time ({timeformat})": x.ravel(), "Wavelength (microns)": y.ravel(), "Flux": rflux.ravel(),
                    "Flux Error": rfluxe.ravel()}
    df = pd.DataFrame(rainbow_dict)
    return df


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
        #   need to add defn for time here?
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
                print(e)
                pass

    bin_t = bins[binned_y != 0]
    bin_e = binned_err[binned_y != 0]
    bin_y = binned_y[binned_y != 0]

    return bin_t, bin_y, bin_e


def find_nearest(array, value):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def remove_nans(arr_with_nans,*otherarrs):
    nanfree_arrs = []
    for arr in otherarrs:
        nanfree_arrs.append(arr[~np.isnan(arr_with_nans)])
    arr_without_nans = arr_with_nans[~np.isnan(arr_with_nans)]
    return arr_without_nans, nanfree_arrs