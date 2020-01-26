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
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as la
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=False)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('figure.subplot', hspace=.4)
# -

# # Simulation of Gaussian processes
# In this section, we will explore two different techniques for the simulation
# of Gaussian processes. The two techniques will be illustrated on a few
# examples: Brownian motion,
# [Brownian bridge](https://en.wikipedia.org/wiki/Brownian_bridge),
# [Fractional Brownian motion](https://en.wikipedia.org/wiki/Fractional_Brownian_motion),
# and [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process).

# +
# Set the final time, number of time steps, and number of replicas.
T, n, m = 2, 100, 50
s = np.linspace(0, T, n)
t = np.linspace(0, T, n)

# Define covariance function of the processes
covariance_functions = {
    'Brownian motion': lambda t, s:
        # np.minimum calculates the element-wise minimum
        np.minimum(t, s),
    'Brownian bridge': lambda t, s, T=T:
        # T is the final time
        np.minimum(s, t) * (T - np.maximum(s, t)) / T,
    'Fractional Brownian motion': lambda t, s, h=.8:
        # h is the Hurst index
        (1/2) * ((t**2)**h + (s**2)**h - ((t - s)**2)**h),
    'Ornstein-Uhlenbeck': lambda t, s, theta=1, mu=-1, sigma=1, x0=-1:
        # theta and mu are the coefficients in the drift, sigma is the
        # diffusion, and x0 is the initial condition.
        sigma**2/(2*theta) * (np.exp(- theta * np.abs(t - s))
          - np.exp(- theta * np.abs(t + s)))
    }

# Mean of the processes (if not zero)
mean_functions = {'Ornstein-Uhlenbeck': lambda t, theta=1, mu=-1, sigma=1, x0=1:
        mu + (x0 - mu) * np.exp(-theta*t)}
# -

# ## Direct simulation

# +
# Generate m samples from the standard mutivariate normal distribution,
# with which we will construct the other processes.
# (Each line is a sample from the Gaussian)
x = np.random.randn(m, n)

# Generate all (s, t) pairs - try this in a REPL to understand what it's doing.
grid_S, grid_T = np.meshgrid(s, t)

fig, ax = plt.subplots(2, 2)
ax = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]

# Sample from each process
for i, process in enumerate(covariance_functions):

    # Get covariance function of the process
    Sigmafunc = covariance_functions[process]

    # Calculate mean of the multivariate Gaussian
    mean = mean_functions[process](t) \
      if process in mean_functions else np.zeros(n)

    # Make the mean a column vector
    mean.shape = (n, 1)

    # Covariance matrix of the multivariate Gaussian
    Sigma = Sigmafunc(grid_S, grid_T)

    # Sigma needs to be positive definite to use the Cholesky factorization.
    # We will therefore use the matrix square root.
    # C = np.linalg.cholesky(Sigma)

    # Calculate factorization of Sigma
    C = np.real(la.sqrtm(Sigma))

    # Samples from the process
    x_process = mean + C.dot(x.T)

    # Plot the process
    ax[i].plot(s, x_process)
    ax[i].set_title(process)
    ax[i].set_xlabel('$t$')

plt.show()
