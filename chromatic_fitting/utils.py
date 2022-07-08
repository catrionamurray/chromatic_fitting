from .imports import *


def noise_calculator(data, maxnbins=None, binstep=1):
    """
    Author: Hannah R. Wakeford, University of Bristol
    Email: hannah.wakeford@bristol.ac.uk
    Citation: Laginja & Wakeford, 2020, JOSS, 5, 51 (https://joss.theoj.org/papers/10.21105/joss.02281)

    Calculate the noise parameters of the data by using the residuals of the fit
    :param data: array, residuals of (2nd) fit
    :param maxnbins: int, maximum number of bins (default is len(data)/10)
    :param binstep: bin step size
    :return:
        red_noise: float, correlated noise in the data
        white_noise: float, statistical noise in the data
        beta: float, scaling factor to account for correlated noise
    """

    # bin data into multiple bin sizes
    npts = len(data)
    if maxnbins is None:
        maxnbins = npts / 10.0

    # create an array of the bin steps to use
    binz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)

    # Find the bin 2/3rd of the way down the bin steps
    midbin = int((binz[-1] * 2) / 3)

    nbins = np.zeros(len(binz), dtype=int)
    standard_dev = np.zeros(len(binz))
    root_mean_square = np.zeros(len(binz))
    root_mean_square_err = np.zeros(len(binz))

    for i in range(len(binz)):
        nbins[i] = int(np.floor(data.size / binz[i]))
        bindata = np.zeros(nbins[i], dtype=float)

        # bin data - contains the different arrays of the residuals binned down by binz
        for j in range(nbins[i]):
            bindata[j] = np.mean(data[j * binz[i] : (j + 1) * binz[i]])

    # get root_mean_square statistic
    root_mean_square[i] = np.sqrt(np.mean(bindata**2))
    root_mean_square_err[i] = root_mean_square[i] / np.sqrt(2.0 * nbins[i])

    expected_noise = (np.std(data) / np.sqrt(binz)) * np.sqrt(nbins / (nbins - 1.0))

    final_noise = np.mean(root_mean_square[midbin:])
    base_noise = np.sqrt(final_noise**2 - root_mean_square[0] ** 2 / nbins[midbin])

    # Calculate the random noise level of the data
    white_noise = np.sqrt(root_mean_square[0] ** 2 - base_noise**2)
    # Determine if there is correlated noise in the data
    red_noise = np.sqrt(final_noise**2 - white_noise**2 / nbins[midbin])
    # Calculate the beta scaling factor
    beta = (
        np.sqrt(root_mean_square[0] ** 2 + nbins[midbin] * red_noise**2)
        / root_mean_square[0]
    )

    # If White, Red, or Beta return NaN's replace with 0, 0, 1
    white_noise = np.nan_to_num(white_noise, copy=True)
    red_noise = np.nan_to_num(red_noise, copy=True)
    beta = 1 if np.isnan(beta) else beta

    # Plot up the bin statistic against the expected statistic
    # This can be used later when we are setting up unit testing.
    plt.figure()
    plt.errorbar(
        binz,
        root_mean_square,
        yerr=root_mean_square_err,
        color="k",
        lw=1.5,
        label="RMS",
    )
    plt.plot(binz, expected_noise, color="r", ls="-", lw=2, label="expected noise")

    plt.title("Expected vs. measured noise binning statistic")
    plt.xlabel("Number of bins")
    plt.ylabel("RMS")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig('noise_calculator.pdf')

    return white_noise, red_noise, beta


def rainbow_to_vector(r, timeformat="h"):
    """Convert Rainbow object to np.arrays
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
    secondformat = ["second", "seconds", "sec", "s"]
    minuteformat = ["minute", "minutes", "min", "m"]
    hourformat = ["hour", "hours", "h"]
    dayformat = ["day", "days", "d"]
    yearformat = ["year", "years", "y"]

    rflux = r.fluxlike[
        "flux"
    ]  # flux (MJy/sr)        : [n_wavelengths x n_integrations]
    rfluxe = r.fluxlike[
        "uncertainty"
    ]  # flux error (MJy/sr)  : [n_wavelengths x n_integrations]
    rtime = r.timelike["time"]  # time (BJD_TDB, hours) : [n_integrations]
    rwavel = r.wavelike["wavelength"]  # wavelength (microns) : [n_wavelengths]

    # change the time array into the requested format (e.g. seconds, minutes, days etc.)
    if timeformat in secondformat:
        rtime = rtime * 3600
    elif timeformat in minuteformat:
        rtime = rtime * 60
    elif timeformat in hourformat:
        # hours is the default time setting
        pass
    elif timeformat in dayformat:
        rtime = rtime / 24.0
    elif timeformat in yearformat:
        rtime = rtime / (24 * 365.0)
    else:
        warnings.warn("Unrecognised Time Format!")
        return

    return rflux, rfluxe, rtime, rwavel


def rainbow_to_df(r, timeformat="h"):
    """Convert Rainbow object to pandas dataframe
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
    rainbow_dict = {
        f"Time ({timeformat})": x.ravel(),
        "Wavelength (microns)": y.ravel(),
        "Flux": rflux.ravel(),
        "Flux Error": rfluxe.ravel(),
    }
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


def remove_nans(arr_with_nans, *otherarrs):
    nanfree_arrs = []
    for arr in otherarrs:
        nanfree_arrs.append(arr[~np.isnan(arr_with_nans)])
    arr_without_nans = arr_with_nans[~np.isnan(arr_with_nans)]
    return arr_without_nans, nanfree_arrs
