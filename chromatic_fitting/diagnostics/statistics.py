from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def chi_sq(data, model, uncertainties, degrees_of_freedom, plot=False):
    # fit_params = len(cmod.summary)
    # degrees_of_freedom = (cmod.data.nwave * cmod.data.ntime) - fit_params

    chi_sq = np.nansum(((data - model) / uncertainties) ** 2)
    red_chi_sq = chi_sq / degrees_of_freedom
    print(f"chi squared = {chi_sq}")
    print(f"Reduced chi squared = {red_chi_sq}")

    # we calculate the p-value using the survival function of the chi^2 distribution
    pvalue = chi2.sf(chi_sq, degrees_of_freedom)

    # we use the "inverse survival function" to determine how many sigma above the mean
    nsigma = norm.isf(pvalue)

    print(
        f"""For {degrees_of_freedom} degrees of freedom, a model that 
    accurately describes the data could result in a 
    $\chi^2>${chi_sq:.3g} with a probability ($p$-value) of {pvalue:.3g}."""
    )

    # depending on whether reduced chi_sq is < or > 1 we change the wording to be more intuitive:
    if nsigma < 0:
        nsigma_phrase = f"{-nsigma:.3g}$\sigma$ below"
    else:
        nsigma_phrase = f"{nsigma:.3g}$\sigma$ above"

    print(
        f"""A model with a $p$-value of {pvalue:.3g} is just as
        unlikely as drawing a value more than {nsigma_phrase} the 
        mean of a Gaussian distribution."""
    )

    if plot:
        # let's plot all this
        x = np.linspace(-7, 7, 1000)
        # probability = norm.sf(x)
        dPdx = norm.pdf(x)

        # plot the probability distribution
        plt.figure()
        plt.plot(x, dPdx, color="black")
        plt.fill_between(x, 0, dPdx, where=(x > nsigma), color="blue", alpha=0.3)
        plt.axvline(nsigma, label=f"{nsigma:.3g}$\sigma$")
        plt.ylabel("dP/dx")
        plt.ylabel(
            """Probability of randomly
        drawing a value $>x$ from 
        a Gaussian distribution"""
        )
        plt.title(
            f"""A model with a $p$-value of {pvalue:.3g} is just as
        unlikely as drawing a value more than {nsigma_phrase} the mean 
        of a Gaussian distribution."""
        )
        plt.legend()

    return chi_sq, red_chi_sq


def generate_periodogram(x, fs, **kw):
    from scipy.signal import periodogram

    freq, power = periodogram(x=x, fs=fs, **kw)
    plt.semilogx(1 / freq, power)
    plt.ylabel("Power")
    plt.xlabel("Time (d)")
    plt.show()
    plt.close()
