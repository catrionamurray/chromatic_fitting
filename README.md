![Testing](https://github.com/catrionamurray/chromatic_fitting/actions/workflows/python-package.yml/badge.svg)
[![GitHub release](https://img.shields.io/github/v/release/catrionamurray/chromatic_fitting?display_name=release&include_prereleases)](https://github.com/catrionamurray/chromatic_fitting/releases/tag/v0)

# chromatic_fitting


This `chromatic_fitting` package is being designed as a companion to [chromatic](https://github.com/zkbt/chromatic). `chromatic_fitting` can perform efficient model fits to spectroscopic light-curve data and produce transmission (or emission) spectra. This package can combine any number of transit, eclipse, polynomial (in time, in x or y position, etc.), Gaussian Process, or user-defined models and fit for all parameters at once. Therefore, in just one fit we can account for the spectral signatures imprinted by planets, stellar activity and instrumental systematics. `chromatic_fitting` also has the flexibility to carry out a 'white light’ fit and perform multi-wavelength fitting (either simultaneously or separately), so we can fully exploit the wide wavelength coverage of different facilities.

This tool was developed alongside the JWST ERS program and successfully applied to the first results from several JWST instruments. However, we see the `chromatic` and `chromatic_fitting` tools as highly applicable to any transit or eclipse observations (including photometric).

# Documentation
Full documentation and tutorials are available [here](https://catrionamurray.github.io/chromatic_fitting/)!

# Installation 
To install this code (at its last stable release version) run:

`pip install git+https://github.com/catrionamurray/chromatic_fitting.git@v0.3.0-stable`

which should install the chromatic_fitting package as well as any necessary dependencies. If you want to install the newest (in development) version then run:

`pip install git+https://github.com/catrionamurray/chromatic_fitting`

If you already have chromatic_fitting already installed but need a newer version then run:

`pip install --upgrade git+https://github.com/catrionamurray/chromatic_fitting`

