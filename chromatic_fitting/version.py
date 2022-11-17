# VERSION HISTORY
# v 0.3.0 - stable release
# v 0.4.0 - changed to vectorized backend
# v 0.5.0 - added step model
# v 0.6.0 - added trapezoid model
#    v 0.6.1 - patch to fix polynomial/linear tutorial fit, fix plot_priors
#    v 0.6.2 - path to add corner plot and add model creation to plot_priors and posteriors
#    v 0.6.3 - trim white light curve

__version__ = "0.6.3"


def version():
    return __version__
