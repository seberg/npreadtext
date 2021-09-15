import numpy as np

from . import _loadtxt

import warnings
warnings.warn(
    "This version of `npreadtext` is meant for testing purposes only; "
    "the proposal is to integrate it into NumPy.\n"
    "Monkeypatching NumPy should not be used for production code.")


np.loadtxt = _loadtxt
np.lib.loadtxt = _loadtxt
np.lib.npyio.loadtxt = _loadtxt
