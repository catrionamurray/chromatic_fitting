# `chromatic_fitting`


The `chromatic_fitting` package is a friendly, open-source python package that is built on the [`chromatic`](https://zkbt.github.io/chromatic/) tool. `chromatic` transforms spectroscopic light curves into `Rainbow` 🌈 objects which allows for easy visualization and comparison.

This package uses [`pymc3`](https://docs.pymc.io/en/v3/index.html) and [`exoplanet`](https://docs.exoplanet.codes/en/latest/) to perform transit fits (as well as fits for other models) to spectroscopic light curve data. It can combine transit models with polynomial models (in time, x, y etc.) or Gaussian Process models to account for the systematics in the light curves and to fit all parameters simultaeously. There is also the option to perform fits for a 'white light curve', fit wavelengths independently or to fit all wavelengths simultaneously. This tool will also produce a transmission spectrum from the chosen fits.

The goal of `chromatic_fitting` is to aid the fast, easy comparison of different data reduction techniques by standardizing the light curve-fitting stage. Its flexibility also allows the comparison of different types of fits and models to understand the impact on the corresponding transmission spectra. 
