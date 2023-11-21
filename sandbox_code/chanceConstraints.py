import numpy as np

import matplotlib.pyplot as plt

from scipy.special import erf, erfinv
from statistics import NormalDist



pd_mean = .2
pd_var = .1

delta = .85

dist = NormalDist(pd_mean, np.sqrt(pd_var))
rho = .5

print(pd_mean-rho)
print(-erfinv(2*delta-1)*np.sqrt(2*pd_var))
print(dist.cdf(rho))