import chromatic
import numpy as np

import exoplanet as xo
import starry
import theano

# this line is important to stop the outdated version of theano messing up starry:
theano.config.gcc__cxxflags += " -fexceptions"

# import theano
# import pickle5 as pickle # needed for Python v3.7!
import sys
from .version import __version__
import matplotlib.pyplot as plt
import pandas as pd
from chromatic import *

import pymc3 as pm
import pymc3_ext as pmx

print(f"Running chromatic_fitting v{__version__}!\n")
print("This program is running on:")
print(f"Python v{sys.version}")
print(f"numpy v{np.__version__}")
print(f"chromatic v{chromatic.version()}")
print(f"pymc3 v{pm.__version__}")
print(f"pymc3_ext v{pmx.__version__}")
print(f"exoplanet v{xo.__version__}")
