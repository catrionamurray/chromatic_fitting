__version__ = "0.9.4"


def version():
    return __version__


# ** VERSION HISTORY **
# v 0.3.0 - stable release
# v 0.4.0 - changed to vectorized backend
# v 0.5.0 - added step model
# v 0.6.0 - added trapezoid model
#    v 0.6.1 - patch to fix polynomial/linear tutorial fit, fix plot_priors
#    v 0.6.2 - path to add corner plot and add model creation to plot_priors and posteriors
#    v 0.6.3 - trim white light curve
# v 0.7.0 - added spot model
# v 0.7.1 - added kw to CombinedModel.setup_lightcurves()
# v 0.7.2 - added some diagnostic tools
# v 0.8.0 - changed pymc3_ext.sample() to pymc3.sample()
# v 0.9.0 - added exponential model
# v 0.9.1 - made t0 a mandatory param of exponential model
# v 0.9.2 - added axis argument to .plot_transmission_spectrum()
# v 0.9.3 - added animated transmission spectrum plot
# v 0.9.4 - summarize now runs automatically in sample, and default HDI range is 16-84
