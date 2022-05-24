import chromatic
import pandas as pd
import numpy as np
import copy
# import exoplanet as xo
# import starry
from exotic_ld import limb_dark_fit
# import theano
from astropy.constants import G
from astropy import units as u
import math
import pickle
# import pickle5 as pickle # needed for Python v3.7!
import src.spectrum
from astropy.table import Table
import warnings
import matplotlib.pyplot as plt
import sys
import time as ttime

print(f"Running on Python v{sys.version}")
print(f"Running on numpy v{np.__version__}")
print(f"Running on chromatic v{chromatic.version()}")
# print("chromatic version = " + str(chromatic.version()))
# print("numpy version = " + str(np.__version__))
# print("theano version = " + str(theano.__version__))