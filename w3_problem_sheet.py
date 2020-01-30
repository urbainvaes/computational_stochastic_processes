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
import time
import numpy as np
import numpy.linalg as la
import scipy.special
import scipy.optimize
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=False)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('figure.subplot', hspace=.4)
# -

# # Generalized Bernoulli distribution

# +
def rand_bernoulli(probs, n):
    cumul = np.cumsum(probs)
    u = np.random.rand(n)
    result = np.ones(n, dtype=int)
    for c in cumul:
        result += (u > c)
    return result

n = 10**3
probs = [.125, .125, .375, .375]
x = rand_bernoulli(probs, n)
pmf = [sum(x == i)/n for i in range(1, len(probs) + 1)]

# Alternative...
# pmf = np.histogram(x, bins=(.5 + np.arange(len(probs))))[0]

print(pmf)
# -
# # Sampling from $\text{Gamma}(k, \lambda)$

gamma = scipy.special.gamma


def rand_gamma_int(lam, k, n):
    u = np.random.rand(k, n)
    return (-1/lam)*np.sum(np.log(u), axis=0)

def rand_cauchy(n):
    u = np.random.rand(n)
    return np.tan(np.pi*(u - .5))

def gamma_pdf(x, lam, k):
    positive, x = x > 0, np.abs(x)
    return positive * ((lam**k)*x**(k-1)*np.exp(-lam*x) / gamma(k))

def cauchy_pdf(x):
    return 1/(np.pi*(1 + x**2))

def timeit(fun):
    def fun_with_timer(*args, **kwargs):
        t0 = time.time()
        result = fun(*args, **kwargs)
        t1 = time.time()
        print("Time elapsed in {}: {}".format(fun.__name__, t1 - t0))
        return result
    return fun_with_timer

@timeit
def gamma_reject(lam, k, n):
    assert k > 1  # We need k > 1 because gamma(0) = ∞
    k0, lam0 = int(k), (int(k)/k)*lam
    x_star = 0 if k0 == k else (k - k0)/(lam - lam0)
    M = lam**k/lam0**k0 * gamma(k0)/gamma(k) * \
        (x_star/np.exp(1))**(k-k0)
    print(M)
    n = int(n*M) + 1  # We do this to generate approximately n samples
    x = rand_gamma_int(lam0, k0, n)
    u = np.random.rand(n)
    indices = M*u <= gamma_pdf(x, lam, k)/gamma_pdf(x, lam0, k0)
    return x[np.where(indices)]

@timeit
def gamma_reject_cauchy(k, n):  # Only for lam == 1
    M = (np.pi/gamma(k))*((k-1)**(k-1)*np.exp(-(k-1))*np.pi +
                           np.pi*(k+1)**(k+1)*np.exp(-(k+1)))
    print(M)
    n = int(n*M) + 1  # We do this to generate approximately n samples
    x = rand_cauchy(n)
    u = np.random.rand(n)
    indices = M*u <= gamma_pdf(x, 1, k)/cauchy_pdf(x)
    return x[np.where(indices)]


n, x = 10**4, np.linspace(0, 10, 500)
def test(k, lam=1, plot=False):
    y1 = gamma_reject(lam, k, n)
    y2 = gamma_reject_cauchy(k, n)
    if plot:
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(x, gamma_pdf(x, lam, k))
        ax[1].plot(x, gamma_pdf(x, lam, k))
        ax[0].hist(y1, bins=20, density=True)
        ax[1].hist(y2, bins=20, density=True)
        plt.show()

test(2.5)
test(4.5)
test(8.5, plot=True)

# def gamma_reject_cauchy(lam, k, n):

# #  Calculating the area of Batman

# +
def batman_indicator(x, y):

    # We'll initialize at one and remove parts one by one
    result = np.ones(x.shape)

    # Ellipse
    ellipse = (x/7)**2 + (y/3)**2 - 1 >= 0
    result[np.where(ellipse)] = 0

    # Bottom curve on [-3, 3]
    bottom = (abs(x) < 4) * \
             (y <= abs(x/2) - ((3*np.sqrt(33)-7)/112)*x**2 - 3
              + np.sqrt(np.maximum(0, 1-(abs(abs(x)-2) - 1)**2)))
    result[np.where(bottom)] = 0

    # Top curve
    top = (abs(x) > .75) * (abs(x) < 1) * (y > 9 - 8*abs(x)) \
          + (abs(x) > .5) * (abs(x) < .75) * (y > 3*abs(x) + .75) \
          + (abs(x) < .5) * (y > 2.25) \
          + (abs(x) > 1) * (abs(x) < 3) * \
            (y > (6*np.sqrt(10)/7+(1.5-.5*abs(x))-(6*np.sqrt(10)/14)*\
                  np.sqrt(np.maximum(0, 4-(abs(x)-1)**2))))
    result[np.where(top)] = 0
    return result

# Exact area
I = (955/48) - (2/7) * (2*np.sqrt(33) + 7*np.pi + 3*np.sqrt(10) * (np.pi - 1)) \
    + 21 * (np.arccos(3/7) + np.arccos(4/7))


xs = np.arange(-7.25, 7.25, 0.01)
ys = np.arange(-3.1, 3.1, 0.01)
x, y = np.meshgrid(xs, ys)
fig, ax = plt.subplots()
ax.contourf(x, y, batman_indicator(x, y))
plt.show()
# -

# +
# Dimensions of the bounding box
Lx, Ly = 7.25, 4

def Monte_Carlo(fun, n=1000):
    x, y = np.random.rand(2, n)
    x, y = Lx * (2*x - 1), Ly * (2*y - 1)
    result = 4*Lx*Ly * fun(x, y)
    return np.mean(result), result
# -

# ## Construction of confidence intervals
# Note that, below, we estimate the probability that $|I - \hat I_n| <
# a_{95\%}$, where $a_{95\%}$ is the half-width of 95% the confidence interval
# constructed by one of the method, also by employing a Monte-Carlo method! In
# particular, we could in principle construct a confidence interval for that
# probability!

# +
# 95% confidence interval using the three methods
alpha = .05

# Calculate half-width (up factor the common factor)
hw_cheb = 1/np.sqrt(alpha)
hw_clt = np.sqrt(2)*scipy.special.erfinv(1 - alpha)
def hw_bikelis_fun(m2, m3, n):
    root = scipy.optimize.root(
        fun = lambda a: scipy.special.erf(a/np.sqrt(2))
                         - 2*m3/(m2**(3/2)*(1 + np.abs(a))**3)/np.sqrt(n)
                         - (1 - alpha),
            x0 = hw_clt)

    if root.status:
        return root.x[0]
    else:
        raise Exception("Root not found!")

# Repeat the Monte Carlo several times to estimate the probability that I is
# in our confidence interval
n_times = 1000

# We will construct confidence intervals based on the sample variance
result_cheb = np.zeros(n_times)
result_clt = np.zeros(n_times)
result_bikelis = np.zeros(n_times)
result_hoeffdings = np.zeros(n_times)

for i in range(n_times):
    In, result = Monte_Carlo(batman_indicator)

    # Number of samples used in MC estimator
    n = len(result)

    # Calculate sample variance
    m2, m3 = np.mean((result - In)**2), np.mean((result - In)**3)

    # Common factor in confidence intervals
    factor = np.sqrt(m2/n)

    # Calculate half-width of the confidence interval with Bikelis
    hw_bikelis = hw_bikelis_fun(m2, m3, n)

    # With Hoeffding's theorem: here no factor sigma/sqrt(n)
    hw_hoeffdings = 4*Lx*Ly * np.sqrt(- np.log((alpha)/2)/(2*n))

    result_cheb[i] = abs(I - In) < hw_cheb * factor
    result_clt[i] = abs(I - In) < hw_clt * factor
    result_bikelis[i] = abs(I - In) < hw_bikelis * factor
    result_hoeffdings[i] = abs(I - In) < hw_hoeffdings

def print_confidence(method, value):
    print("(Approximate) actual confidence of the 95% conf. int. "
          "constructed via {}: {:.4f} ".format(method, value))

print_confidence("Chebyshev's inequality", np.mean(result_cheb))
print_confidence("the CLT", np.mean(result_clt))
print_confidence("Bikelis' theorem", np.mean(result_bikelis))
print_confidence("Hoeffdings' theorem", np.mean(result_hoeffdings))
# -
# Variance reduction by variate

# +
def ellipse_indicator(x, y):
    return ((x/7)**2 + (y/3)**2 - 1 < 0)*1.

# Area of ellipse
E = np.pi * 7 * 3

# Monte Carlo estimation
n = 10**6
x, y = np.random.rand(2, n)
x, y = Lx * (2*x - 1), Ly * (2*y - 1)
result_batman = 4*Lx*Ly * batman_indicator(x, y)
result_ellipse = 4*Lx*Ly * ellipse_indicator(x, y)

# Estimate the optimal coefficient
Sigma = np.cov(result_batman, result_ellipse)
alpha = - Sigma[0, 1] / Sigma[1, 1]

# Estimator with control variate
result_c = result_batman + alpha * (result_ellipse - E)

# Variance (times n)
var_c = np.var(result_c)

print("Variance (times n) without and with control variate:",
      "{:.2f}, {:.2f}.".format(Sigma[0, 0], var_c))

# Compare with expected variance reduction
exact_gain = 1 - (I*(4*Lx*Ly -E))/(E*(4*Lx*Ly - I))
print("Observed and exact variance reduction:",
      "{:.3f}, {:.3f}.".format(var_c/Sigma[0, 0], exact_gain))

## Gambler's ruin

# +
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
          "[{0:0.6f} - {1:0.2e}, {0:0.6f} + {1:0.2e}]"
          .format(m, a*np.sqrt(v)))

n = 10**6

# Without importance sampling
mean, var = gamblers_ruin(b_fun=lambda x: 0, n=n)
print_confidence(mean, var/n)

# With basic importance sampling
mean_is, var = gamblers_ruin(b_fun=lambda x: 0*x + b, n=n)
print_confidence(mean, var/n)

# With improved importance sampling
mean_im, var = gamblers_ruin(b_fun=lambda x: (x > 0)*b, n=n)
print_confidence(mean_im, var/n)
