import numpy as np
# try:
#     np.distutils.__config__.blas_opt_info = np.distutils.system_info.blas_opt_info#np.__config__.blas_ilp64_opt_info
# except Exception:
#     pass
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