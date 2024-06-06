__version__ = "0.12.15"


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
#    v 0.7.1 - added kw to CombinedModel.setup_lightcurves()
#    v 0.7.2 - added some diagnostic tools
# v 0.8.0 - changed pymc3_ext.sample() to pymc3.sample()
# v 0.9.0 - added exponential model
#    v 0.9.1 - made t0 a mandatory param of exponential model
#    v 0.9.2 - added axis argument to .plot_transmission_spectrum()
#    v 0.9.3 - added animated transmission spectrum plot
#    v 0.9.4 - summarize now runs automatically in sample, and default HDI range is 16-84
#    v 0.9.5 - vectorized nsigma parameter (for inflating uncertainties)
#    v 0.9.6 - added setup_lightcurves under setup_likelihood
#    v 0.9.7 - added a simple check for attaching Rainbow objects and setting up WLC (nw>0, nt>0)
#    v 0.9.8 - fixed exponential bug
# v 0.10.0 - added several features from restructure branch and new tests. Added xlim arg.
#   v 0.10.1 - added eclipse model
#       v 0.10.1.2 - added some helper functions
# v 0.11.0 - added spot model
#    v 0.11.1 - added multiple spot model
#    v 0.11.2 - added overwrite kw to summarize() to allow az.summary() kws to be changed
#    v 0.11.3 - fixed bug in plot_priors and plot_posteriors
# v 0.12.0 - added phase curve model
#    v 0.12.1 - added GP model
#    v 0.12.2 - fixed bug in phase curve + eclipse model building
#    v 0.12.9 - changed plot_priors and plot_posteriors to imshow
#    v 0.12.10 - added edge case for fixed deterministic param
#    v 0.12.11 - fixed duplicate name issue
#    v 0.12.14 - added Matern-3/2 kernel to GP model
#    v 0.12.15 - added .plot_initial_guess() func and fixed some docs