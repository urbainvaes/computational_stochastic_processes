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
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib (for plots)
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=4)


# -

# We start by implementing the Linear Congruential Generator with the default parameters given in the lecture notes.

def lcg(n, x0, M=2**32, a=22695477, c=1):
    """ Generate pseudo-random numbers with the LCG method

    The LCG is based on the iteration

        x(n+1) = (a*x(n) + c) % M

    Parameters
    ----------
    n : integer
        The number of pseudo-random variables to generate
    x0 : integer
        The seed
    M : integer
        The modulus
    a : integer
        The multiplier
    c : integer
        The increment   

    Returns
    -------
    A Numpy array of `n` pseudo-random varibales

    Note
    ----
    The default parameters are the ones used by glibc

    """

    result = np.zeros(n)
    for i in range(n):
        x0 = (a*x0 + c) % M
        result[i] = x0/float(M)

    return result


# Let us generate $10^5$ random variables using our random number generator.

# With glibc parameters (good generator)
x = lcg(10**5, 3)

# Although there is no universal and foolproof test that can guarantee that a RNG or PRNG is good, in practice a number of tests have been developed to detect whether a simulation method is bad: the Kolmogorov-Smirnov test (see below), the $\chi^2$ test, etc. If these tests fail, we can reject the hypothesis that the numbers produced were drawn from a uniform distribution.
#
# Before presenting the Kolmogorov-Smirnov test, let us check that the empirical PDF of our data is close to the expected one.

# +
fig, ax = plt.subplots(1, 2)

# Number of points for the plots
n_grid = 200

# Plot histogram (an approximation of the PDF) and the expected PDF
u = np.linspace(0, 1, n_grid)
exact_pdf = np.ones(n_grid)
ax[0].hist(x, bins=20, density=True)
ax[0].plot(u, exact_pdf)
ax[0].set_title("Histogram and exact PDF")


# Pair the values in the array x 2 by 2, calculate the difference between
# the elements of each pair, and plot the results in a histogram.
#
# Note: the difference of two uniformly-distributed random
# variables has PDF (1 - |x|) on [-1, 1].
x_odd = x[0::2]
x_even = x[1::2]
u = np.linspace(-1, 1, n_grid)
exact_pdf = (1 - abs(u))
ax[1].hist(x_odd - x_even,bins=20, density=True)
ax[1].plot(u, exact_pdf)
ax[1].set_title("Histogram and exact PDF of the difference")
plt.show()
# -

# # Kolmogorov-Smirnov test
#
# Consider the empirical CDF $F_N$ associated to $N$ random uniformly-distributed samples:
# $$
# F_N(x) = \frac{1}{N} \sum_{i=1}^N I_{(-\infty, x]}(X_i)
# $$
# where $I_{(-\infty, x]}$ is the indicator function of the interval $(-\infty, x]$. 
# By the law of large numbers, for all $x$ it holds that
# $$
# F_N(x) = \frac{1}{N} \sum_{i=1}^N I_{(-\infty, x]}(X_i) \xrightarrow{\text{a.s. as }N \to \infty} \mathbb E(I_{(-\infty, x]}(X_i)) = \mathbb P(X_i \leq x) = F(x) := \max(0, \min(1, x)),
# $$
# where $F$ is the CDF of the uniform distribution.
#
# In fact, we can show more
#
# - *Glivenko-Cantelli theorem*: $D_N := \sup_{x \in \mathbb R} |F_N(x) - F(x)| \to 0$ almost surely as $N \to \infty$.
#
# - *Kolmogorov theorem*: $\sqrt{n} D_N \to K$ in distribution, where $K$ is distributed according to the Kolmogorov distribution.  The CDF of $K$ is given by:
# $$
# \mathbb{P}(K\leq x)=1-2\sum_{k=1}^\infty (-1)^{k-1} e^{-2k^2 x^2}.
# $$

# +
# Sort random samples and calculate the CDF
x = np.sort(x)
cdf = np.arange(1, len(x) + 1)/len(x)
cdf_shifted = np.arange(0, len(x))/len(x)

fig, ax = plt.subplots(1, 2)
# ax[0].hist(x, cumulative=True, bins=100, density=True, histtype='step')
ax[0].plot(x, cdf)
ax[0].plot(x, x)
ax[0].set_title("Empirical and exact CDFs")
ax[1].plot(x, cdf - x)
ax[1].plot(x, 0*x)
ax[1].set_title("Difference between empirical and exact CDFs")
plt.show()
# -

# Now let us calculate the sup norm of the difference between the empirical CDF, 
# based on the data in `x`, and the exact CDF of the uniform distribution.
error_sup = max(np.max(abs(cdf - x)), np.max(abs(cdf_shifted - x)))
normalized_statistic = np.sqrt(len(x)) * error_sup

# Now we calculate approximately the probability (`pvalue` below) of observing `error_sup` greater than or equal to  what we observed when assuming that the
# elements of `x` are drawn from a uniform distribution. This is an approximation because, for finite $N$, our test statistic is not exactly distributed according to the Kolmogorov distribution.
#
# Below we also check that our results are consistent with those obtained by an application of the function `kstest` in `scipy.stats`.
#

# +
def pvalue_kolmogorov(y, truncation=10**5):
    """ Calculate the probability that K ≥ y, if K follows the Kolmogorov
    distribution

    y : float
    truncation: integer
        index at which the series is truncated
    """
    return 2*np.sum([(-1)**(k-1)*np.exp(-2*k**2*y**2)
                     for k in range(1, truncation)])


# We know that, asymptotically, `error_sup` follows the Kolmogorov
# distribution.
pvalue = pvalue_kolmogorov(normalized_statistic)
print("Pvalue calculated: {}".format(pvalue))

# Check that we obtain the correct results
statistic, pvalue = stats.kstest(x, 'uniform', mode='asymp')
print("Pvalue calculated with SciPy: {}".format(pvalue))
# -

# Based on these results, we can't reject the hypothesis that our samples were drawn from a true uniform distribution. The Mersenne-Twister algorithm gives similar results:

np.random.seed(0)
x = np.random.rand(10**5)
statistic, pvalue = stats.kstest(x, 'uniform', mode='asymp')
print("Pvalue calculated with SciPy: {}".format(pvalue))

# Interpretation: `pvalue` is the probablility of observing `error_sup`
# greater than or equal to that what we observed when assuming that the
# elements of `x` are drawn from a uniform distribution.
