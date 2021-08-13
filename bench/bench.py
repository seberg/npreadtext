import numpy as np
import pandas as pd
from npreadtext import _loadtxt

# Prepare csv file
a = np.random.rand(100_0000, 5) * 100000
np.savetxt("test.csv", a, delimiter=",", fmt="%d")

cmds_to_bench = [
    #'b = np.loadtxt("test.csv", delimiter=",")',
    #'c = pd.read_csv("test.csv", delimiter=",")',
    #'d = pd.read_csv("test.csv", delimiter=",", float_precision="round_trip")',
    'e = _loadtxt("test.csv", delimiter=",")',
]


for i in range(10000):
    _loadtxt("test.csv", delimiter=",", dtype=np.int64)

#for i in range(20):
#    for cmd in cmds_to_bench:
#        print(f">>> {cmd}")
#        get_ipython().run_line_magic('timeit', cmd)
