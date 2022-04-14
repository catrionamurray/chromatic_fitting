import chromatic
import numpy as np
import pandas as pd
import numpy as np
import exoplanet as xo
import starry
from exotic_ld import limb_dark_fit
# import theano
from astropy.constants import G
from astropy import units as u
import math
import pickle
import spectrum
from astropy.table import Table

print("chromatic version = " + str(chromatic.version()))
print("numpy version = " + str(np.__version__))
# print("theano version = " + str(theano.__version__))