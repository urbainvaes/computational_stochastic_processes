# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#+
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import math
import numpy as np
import numpy.linalg as la
import scipy.special
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
# -

def gamblers_ruin(b_fun, n, plot=False):

    # Parameters
    s0, sigma, N = 1, .1, 10

    # Target function to integrate
    def f(x):
        return (np.min(x, axis=0) <= 0)*1.

    # Likelihood ratio
    def g(x):

        n_paths = x.shape[1]

        result = np.zeros(n_paths)
        for i in range(N):
            bi = b_fun(x[i, :])
            result += bi * (x[i + 1, :] - x[i, :]) - (1/2) * bi**2

        return np.exp(-(1/sigma**2) * result)

    n_per_slice = 10**6
    n_slices = n // n_per_slice
    n = n_per_slice * n_slices
    mn, qn = 0, 0

    for i in range(n_slices):
        # x = samples from the nominal distribution
        # y = samples from the importance distribution
        z = sigma * np.random.randn(N, n_per_slice)

        # We store the initial condition in x too
        x = np.zeros((N + 1, n_per_slice))

        # Set initial condition
        x[0, :] = s0

        for j in range(N):
            x[j + 1] = x[j] + b_fun(x[j]) + z[j]

        fx, gx = f(x), g(x)
        mn = 1/(i+1) * (mn*i + np.mean(fx*gx))
        qn = 1/(i+1) * (qn*i + np.mean((fx*gx)**2))

    if plot:
        n_samples = 20
        fig, ax = plt.subplots()
        ax.plot(np.arange(N + 1), x[:, :n_samples], marker='.')
        ax.plot(np.arange(N + 1), np.zeros(N + 1), linestyle='--', color='k')
        ax.set_xlabel("$k$")
        plt.show()

    return mn, n/(n - 1) * (qn - mn**2)

# Default parameter
b = -.1

def print_confidence(m, v):
    a = scipy.stats.norm.ppf(.975)
    # or 'manually'...
    # a = np.sqrt(2)*scipy.special.erfinv(.95)
    print("95% confidence interval for the probability of ruin: "
          "[{0:0.6f} - {1:0.2e}, {0:0.6f} + {2:0.2e}]"
          .format(m, a*np.sqrt(v), - a*np.sqrt(v)))


# Without importance sampling
n_not_precise = 10**6
mean, var = gamblers_ruin(b_fun=lambda x: 0, n=n_not_precise)
print_confidence(mean, var/n_not_precise)

# With basic importance sampling
mean, var = gamblers_ruin(b_fun=lambda x: 0*x + b, n=n_not_precise)
print_confidence(mean, var/n_not_precise)

# With improved importance sampling
n_precise = 10**7
mean, var = gamblers_ruin(b_fun=lambda x: (x > 0)*b, n=n_precise)
print_confidence(mean, var/n_precise)
