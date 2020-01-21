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

# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.stats as stats

# Configure matplotlib (for plots)
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=4)
matplotlib.rc('lines', marker='.')
matplotlib.rc('lines', markersize=10)
# -

# # Inverse method
#
# Assume that we want to generate random numbers with PDF
# $$
# f(x) =
# \begin{cases}
#     1/2 & \text{if $x \in [0, 1]$} \\
#     1/2 & \text{if $x \in [2, 3]$} \\
#     0 & \text{otherwise}
# \end{cases}
# $$
#
# We calculate that
# $$
# F(x) =
# \begin{cases}
#     0  & \text{if $x < 0$} \\
#     x/2 & \text{if $x \in [0, 1]$} \\
#     1/2 & \text{if $x \in (1, 2)$} \\
#     x/2 - 1/2 & \text{if $x \in [2, 3]$} \\
#     1  & \text{if $x > 3$} \\
# \end{cases}
# $$
# so
# $$
# G(u) =
# \begin{cases}
#     2u  & \text{if $u \leq 1/2$} \\
#     2u + 1 & \text{if $x > 1/2$} \\
# \end{cases}
# $$
#

# +
def G(u):
    return 2*u + (u > 1/2)

# Number of samples to generate
n = 10**6

# Array of uniformly distributed samples
u = np.random.rand(n)

# Application of the inverse method: the CDF of x should be F
x = G(u)

# Exact PDF and CDF
def f(x):
    return 1/2*(x >= 0)*(x <= 1) + 1/2*(x >= 2)*(x <= 3)

def F(x):
    return (1/2) * np.maximum(0, np.minimum(1, x)) \
         + (1/2) * np.maximum(0, np.minimum(1, x - 2))

# Plot the empirical PDFs and CDFs
fig, ax = plt.subplots(1, 2)
x_plot = np.linspace(0, 3, 400)
ax[0].hist(x, bins=100, density=True)
ax[0].plot(x_plot, f(x_plot))
ax[0].set_title("Histogram and exact PDF")
ax[0].set_xlabel("$x$")

x = np.sort(x)
cdf = np.arange(1, len(x) + 1)/len(x)
ax[1].plot(x, cdf)
ax[1].plot(x, F(x))
ax[1].set_title("Empirical and exact CDFs")
ax[1].set_xlabel("$x$")
plt.show()
# -

# # Rejection sampling method
#
# From the previously generated random numbers, we would now like to generate
# random numbers with PDF
# $$
# h(x) =
# \begin{cases}
#     1/3 & \text{if $x \in [0, 1]$} \\
#     2/3 & \text{if $x \in [2, 3]$} \\
#     0 & \text{otherwise}
# \end{cases}
# $$
#
# Clearly $h(x) ≤ (4/3) f(x)$, so the condition of rejection sampling holds with M = 4/3.
# We calculate
# $$
# \frac{h(x)}{f(x)} = \frac{2}{3} I_{[0, 1]}(x) + \frac{4}{3} I_{[2, 3]}(x).
# $$

# +
# We start by generating the uniform random numbers that will determine whether
# to accept or reject the samples.
u = np.random.rand(n)

# Exact PDF and CDF of our target distribution
def h(x):
    return 1/3*(x >= 0)*(x <= 1) + 2/3*(x >= 2)*(x <= 3)

def H(x):
    return (1/3) * np.maximum(0, np.minimum(1, y)) \
         + (2/3) * np.maximum(0, np.minimum(1, y - 2))

# Now we construct samples with PDF h(.), stored in array `y` below.
# Here we use `x` calculated above.
M = 4/3
y = x[np.where(u < h(x)/(M*f(x)))]

# Plot a histogram and the empirical CDF
fig, ax = plt.subplots(1, 2)
y_plot = np.linspace(0, 3, 400)
ax[0].hist(y, bins=100, density=True)
ax[0].plot(y_plot, h(y_plot))
ax[0].set_title("Histogram and exact PDF")
ax[0].set_xlabel("$y$")

y = np.sort(y)
cdf = np.arange(1, len(y) + 1)/len(y)
ax[1].plot(y, cdf)
ax[1].plot(y, H(y))
ax[1].set_title("Empirical and exact CDFs")
ax[1].set_xlabel("$y$")
plt.show()
# -

# Frequency of acceptance
print("Frequency of acceptance: {}".format(len(y)/len(x)))
print("We expected: {}".format(1/M))

# # Other example of rejection sampling
#
# Now suppose that we are given a generator of standard Gaussian random
# variables and that we would like to generate samples with PDF
# $$
#     f(x) = I_{[0, 1]}(x) \, (.9 + .2 x)
# $$
# Below $g(x)$ denotes the PDF of the standard Gaussian.

# +
# Gaussian PDF
def g(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)

# Target PDF
def f(x):
    return (x >= 0)*(x <= 1)*(.9 + .2*x)

# Array of elements drawn from N(0, 1)
x = np.random.randn(n)

# Best possible M (Why?)
M = 1.1/g(1)

# Rejection step
u = np.random.rand(n)
y = x[np.where(u < f(x)/(M*g(x)))]

# Plot the empirical PDFs and CDFs
fig, ax = plt.subplots(1, 2)
y_plot = np.linspace(-4, 4, 400)
ax[0].plot(y_plot, M*g(y_plot), label="$M g(x)$")
ax[0].plot(y_plot, f(y_plot), label="$f(x)$")
ax[0].legend()
ax[0].set_title("Comparison of the two PDFs")

y_plot = np.linspace(-.1, 1.1, 400)
ax[1].hist(y, bins=20, density=True)
ax[1].plot(y_plot, f(y_plot))
ax[1].set_title("Empirical and exact PDFs")
plt.show()
# -

# Frequency of acceptance
print("Frequency of acceptance: {}".format(len(y)/len(x)))
print("We expected: {}".format(1/M))

# In order to strengthen our understanding the proofs related to rejection sampling,
# it is instructive to plot the pairs $(x, M g(x) u)$, where x and u are the normal
# and associated uniform samples employed in the rejection method, respectively.

# +
x = np.random.randn(2*10**3)
u = np.random.rand(2*10**3)

accept = u < f(x)/(M*g(x))
i1 = np.where(accept)
i2 = np.where(np.invert(accept))

fig, ax = plt.subplots()
ax.plot(x[i1], M*g(x)[i1]*u[i1], linestyle='', color='red')
ax.plot(x[i2], M*g(x)[i2]*u[i2], linestyle='', color='blue')

x = np.sort(x)
ax.plot(x, M*g(x), label="$M g(x)$")
ax.plot(x, f(x), label="$f(x)$")
ax.legend()
plt.show()


# -

# # Box-Muller algorithm

# +
def box_muller(n):
    u_1 = np.random.rand(n)
    u_2 = np.random.rand(n)

    X = np.sqrt(-2*np.log(u_2)) * np.cos(2*np.pi*u_1)
    Y = np.sqrt(-2*np.log(u_2)) * np.sin(2*np.pi*u_1)

    return X, Y


X, Y = box_muller(n)
fig, ax = plt.subplots(1, 2)
x_plot = np.linspace(-4, 4, 400)

ax[0].hist(X, bins=30, density=True)
ax[0].plot(x_plot, g(x_plot))
ax[0].set_title("Histogram and exact PDF")
ax[0].set_xlabel('$X$')

ax[1].hist(Y, bins=30, density=True)
ax[1].plot(x_plot, g(x_plot))
ax[1].set_title("Histogram and exact PDF")
ax[1].set_xlabel('$Y$')
plt.show()


# -

# # Generate multivariate Gaussians

# +
def gaussian(mean, covariance, n):
    dim = len(covariance)
    # Transform mean into a column vector
    mean.shape = (dim, 1)
    x = np.random.randn(dim, n)
    sqrt_cov = la.sqrtm(covariance)
    result = np.tile(mean, n) + sqrt_cov.dot(x)
    return result

n = 10**4
mean = np.array([0, 1])
covariance = np.array([[1, .3], [.3, 2]])
x = gaussian(mean, covariance, n)

# Check mean and covariance
print(np.mean(x, axis=1))
print(np.cov(x))
# -

# Plot (you do not need to understand precisely how this works)
fig, ax = plt.subplots()
ax.hist2d(x[0,:], x[1,:], bins=50)
# kernel = stats.gaussian_kde(x)
# grid = np.linspace(-1, 1, 200)
# X, Y = np.meshgrid(3*grid, 1+3*grid)
# positions = np.vstack([X.ravel(), Y.ravel()])
# Z = np.reshape(kernel(positions).T, X.shape)
# ax.set_aspect(1)  # Ensure that the aspect ratio is correct
# ax.contourf(X, Y, Z, levels=60)
plt.show()

# Note that there are many ways of solving the equation $C C^T = \Sigma$ (see lecture notes).
