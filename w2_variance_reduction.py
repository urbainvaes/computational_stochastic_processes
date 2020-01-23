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

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=False)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)

# Fix the seed
np.random.seed(0)
# -


# # Antithetic variables
#
# With this method, the estimator reads
# $$
# \newcommand{\e}{\mathrm{e}}
# \newcommand{\d}{\mathrm{d}}
# \hat I_n^a = \frac{1}{2n} \sum_{i = 1}^n (f(X_i) + f(X_i^a)).
# $$
# A necessary condition for this estimator to be unbiased is that $\mathbb
# E(f(X_i^a)) = I$, which restricts the choice of $X_i^a$.
# In addition, the quality of an antithetic variable depends on the function
# $f$ that we want to integrate.
# Indeed the method leads to a reduction of the variance (at a given
# computational cost) if
# $$
# \textrm{Cov}(f(X_i), f(X_i^a)) < 0.
# $$
# Assume that we want to calculate the integral
# $$
# I = \int_0^1 f(x) \, \mathrm{d} x.
# $$
# A standard choice of antithetic variable when $X_i \sim U(0, 1)$ is $X_i^a = 1 - X_i$.
# But whether this choices leads to a reduction in the variance or not depends on the function $f$.
# Below we consider the choices $f = x + e^x$ and $f = |x - .5|$ and
# we calculate $C_n \textrm{var} \hat I_n$ as well as $C_n \textrm{var} \hat I_n^a$,
# where $C_n$ is the number function evaluations required for the computation of each estimator.

# +
f1 = lambda x: x + np.exp(x)
f2 = lambda x: np.abs(x - .5)

# Number of samples
n = 10**6

# Estimation of the "variance times function evaluations" (which is constant
# with the number of function evaluations) without variance reduction
u = np.random.rand(n)
var_no_reduction = np.var(f1(u))

# Estimation of the variance with antithetic variables.
# We include a factor 2 because each iteration requires 2 function evaluations.
u_a = 1 - u
var_antithetic = 2 * np.var((f1(u) + f1(u_a))/2)
print(var_no_reduction, var_antithetic)
# -

# With $f(x) = x + e^x$, the variance reduction is significant.
# This is not the case for $f(x) = |x - .5|$.

var_no_reduction = np.var(f2(u))
var_antithetic = 2 * np.var((f2(u) + f2(u_a))/2)
print(var_no_reduction, var_antithetic)

# The variance (times the number of function evaluations) of our antithetic estimator is now twice as large as
# that of the unmodified estimator! See if you can understand why?
#
# In general, a sufficient condition under which this approach, based on the
# antithetic variable $X_i^a = 1 - X_i$, leads to a reduction of the variance
# is that **$f(x)$ is a monotonic function**, because the sign of the
# covariance is preserved under monotonic functions.
#
# This approaches generalizes naturally to several dimensions.

# +
def f_circle(x):
    return 4*(x[0]**2 + x[1]**2 < 1)

# On [0, 1] x [0, 1]
u = np.random.rand(2, n)
u_a = 1 - u
evaluations_no_reduction = f_circle(u)
evaluations_antithetic = (f_circle(u) + f_circle(u_a))/2
print(np.mean(evaluations_no_reduction), np.mean(evaluations_antithetic))
print(np.var(evaluations_no_reduction), 2 * np.var(evaluations_antithetic))
# Good!
# -

# +
# On [-1, 1] x [-1, 1]
u = np.random.rand(2, n)
u_a = - u
evaluations_no_reduction = f_circle(u)
evaluations_antithetic = (f_circle(u) + f_circle(u_a))/2
print(np.mean(evaluations_no_reduction), np.mean(evaluations_antithetic))
print(np.var(evaluations_no_reduction), 2 * np.var(evaluations_antithetic))
# Bad! (See why there is a factor two here?)

# -
# # Control variates
#
# Let us assume that we want to caculate
# $$
# I = \int_0^1 e^x \, \mathrm{d} x,
# $$
# and that we know how to integrate monomials:
# $$
# \int_0^1 h_i(x) \, \mathrm{d} x := \int_0^1 x^i \, \mathrm{d} x = \frac{1}{i+1} =: m_i.
# $$
# The $h_i$ here are our **control variates**. Notice that
# $$
# I = \int_0^1 e^x + \sum_{i = 1}^n \alpha_i (h_i(x) - m_i) \, \mathrm{d} x,
# $$
# so we can estimate $I$ by applying the regular Monte Carlo method to $g(x) = e^x + \sum_i \alpha_i (h_i(x) - m_i)$,
# but how do we choose $\alpha_i$ to obtain the best variance reduction?

# +
# First attempt: using the coefficients from Taylor's expansion
def f(x):
    return np.exp(x)

def g(x, n=2):
    result = np.exp(x)
    for i in range(1, n + 1):
        result -= (x**i - 1/(i+1)) / math.factorial(i)
    return result

u = np.random.rand(n)
fu, gu, gu5 = f(u), g(u), g(u, n=5)
print(np.mean(fu), np.mean(gu), np.mean(gu5))
print(np.var(fu), np.var(gu), np.var(gu5))
# -

# The variance reduction is substantial, but is our choice of the coifficients
# optimal? To answer the question, let us calculate
# $$
# F_i = \textrm{cov}(e^U, h_j(U)) = \int_0^1 e^x \, x^{i} \, \mathrm d x - I \, m_i.
# $$
# $$
# M_{ij} = \textrm{cov}(h_i(U), h_j(U)) = \int_0^1 x^{i + j} \, \mathrm d x- m_i \, m_j = \frac{1}{i + j + 1} - \frac{1}{(i + 1)(j + 1)}.
# $$
# Note that we are "cheating" here, because we are using the value of $I$ in order to determine the optimal coefficients.
# To calculate $F_i$, we use the recurrence formula
# $$
# \int e^x \, x^{i} \, \mathrm d x = e^x \, x^i - i \int e^x \, x^{i - 1} \, \mathrm d x
# $$
# Below we neglect, in our computation of the "variance times function
# evaluations", the computational cost associated with the evaluation of the
# cnotrol variates.

# +
def h(x, n=2):
    integral = np.exp(1) - 1
    F = np.zeros(n)
    M = np.zeros((n, n))
    for i in range(n):
        integral = np.exp(1) - (i + 1)*integral
        F[i] = integral - (np.exp(1) - 1)/(i + 2)
        for j in range(n):
            M[i][j] = 1/(i + j + 3) - 1/(i + 2)/(j + 2)

    alphas = la.solve(M, F)
    result = np.exp(x)
    for i in range(1, n + 1):
        result -= (x**i - 1/(i+1)) * alphas[i - 1]
    return result

# We obtain an even better variance reduction!
print(np.mean(h(u)), np.mean(h(u, n=5)))
print(np.var(h(u)), np.var(h(u, n=5)))
# -

# This approaches generalizes naturally to several dimensions.
# Let $C$ denote the circle and $B = \{x, y: |x| + |y| ≤ \sqrt{2}\}$.
# Here we will employ the control variate $I_B(x, y)$ and estimate the optimal
# coefficient numerically.

# +
def f_B(x):
    return (np.abs(x[0]) + np.abs(x[1]) < np.sqrt(2))*1.0

# Expectation of f_B(U_1, U_2)
Eb = 1 - (2 - np.sqrt(2))**2/2

n = 10**5
u = -1 + 2*np.random.rand(2, n)
f_circle_u, f_B_u = f_circle(u), f_B(u)
cov = np.cov(f_circle_u, f_B_u)
alpha = - cov[0][1] / cov[1][1]

def h(x):
    return f_circle(x) + alpha*(f_B(x) - Eb)

print(np.mean(f_circle(u)), np.mean(h(u)))
print(np.var(f_circle(u)), np.var(h(u)))
# -

# # Variance reduction by conditioning
#
# In practice, this technique often amounts to calculating some integrals in a
# multiple integral analytically, and to use regular Monte Carlo on the
# integrand of the result. Consider the problem of calculating
# $$
# I = \int_0^1 \int_0^1 e^{-x^3} (y + 1) + \ln(1 + x^2) \, y^2 \, \d y \, \d x.
# $$
# For fixed $x$, the integrand is polynomial in $y$, so we can easily calculate
# the inner integral analytically.
# $$
# I = \int_0^1 \frac{3}{2} \, e^{-x^3} + \frac{1}{3} \ln(1 + x^2) \, \d x.
# $$

# +
f = lambda x, y: np.exp(-x**3) * (y + 1) + np.log(1 + x**2) * y**2
g = lambda x: (3/2) * np.exp(-x**3) + (1/3) * np.log(1 + x**2)

n = 10**6
x = np.random.rand(2, n)

# Regular Monte Carlo
fxy = f(x[0], x[1])
print(np.mean(fxy), np.var(fxy))

# Variance reduction by conditioning
gx = g(x[0])
print(np.mean(gx), np.var(gx))
# -

# # Importance sampling
# ## Exponential Tilting
# Assume that we want to calculate $\mathbb P(X > x_0)$ for some large $x-0$,
# and that $X \sim \mathcal N(\mu, \sigma)$. A common choice in this case (see
# lecture notes) is to choose
# $$
# \psi(x) = \frac{\pi(x) \, \e^{tx}}{\int_{-\infty}^{\infty} \pi(x) \, \e^{tx} \, \d x },
# $$
# which is exactly the density of $\mathcal N(\mu + t \sigma^2, \sigma^2)$.

# +
# Parameters
n, mu, sigma, x0 = 10**6, 1, 1, 4

# Heuristic choice (not optimal!)
t = (x0 - mu)/sigma**2

# Samples from nominal distribution
x = mu + sigma * np.random.randn(n)

# Samples from the importance distribution
y = mu + t*sigma**2 + sigma * np.random.randn(n)

# Likelihood ratio
g = lambda x: np.exp(- t * (x - mu - t*sigma**2/2))

# Target function
f = lambda x: (x > x0)*1.0  # True * 1.0 = 1.0

# Exact value
exact = 1 - (1/2) * (1 + scipy.special.erf((x0 - mu)/(sigma*np.sqrt(2))))

# Estimators
fx, fy, gy = f(x), f(y), g(y)
print(exact, np.mean(fx), np.mean(fy*gy))
print(np.cov(fx), np.cov(fy*gy))
# -

# ## Integrating a bimodal function
# Here we assume that we want to calculate the integral
# $$
# \int_{\infty}^{\infty} e^{-\beta \, V(x)} \, \gamma(x) \, \mathrm d x,
# $$
# where $\gamma$ is the PDF of $\mathcal N(0, \sigma^2)$ and $V$ is a bistable potential
# $$
# V(x) = \frac 14 (x^2 - a^2)^2.
# $$
# To this end, we will use importance sampling with an importance distribution
# given by a Gaussian mixture
# $$
# \psi(x) = \lambda \, g_{\mu_1,\sigma_1^2}(x) + (1 - \lambda) \, g_{\mu_2, \sigma_2^2}(x), \qquad 0 \leq \lambda \leq 1.
# $$

# +
# Paramters
np.random.seed(0)
sigma, a, beta = 5, 2, 2

# Plots
x_plot = np.linspace(-2*a, 2*a, 200)

# Function of which we want to calculate the expectation
V = lambda x: (1/4)*(x**2 - a**2)**2
f = lambda x: np.exp(-beta*V(x))

def gaussian(mu, sigma):
    return lambda x: 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x - mu)**2/(2*sigma**2))

# Nominal distribution
pi = gaussian(0, sigma)

# Calculate the exact value of the integral
infinity = 100
integral = scipy.integrate.quad(lambda x: f(x)*pi(x), - infinity, infinity)
print("The exact value of the integral is {}, \nwith absolute error less than {}".
      format(integral[0], integral[1]))

# Calculate the variance (times n) using usual MC simulation
n = 10**6
x = sigma * np.random.randn(n)
fx = f(x)

# We'll choose psi to be the "sum" of two gaussians
s = 1
g1, g2 = gaussian(-a, s), gaussian(a, s)
psi = lambda x: (1/2)*g1(x) + (1/2)*g2(x)

# Plot the function to integrate wrt pi and psi
fig, ax = plt.subplots()
ax.plot(x_plot, f(x_plot), label=r"$f(x)$")
ax.plot(x_plot, f(x_plot) * pi(x_plot) / psi(x_plot),
        label=r"$f(x) \, \pi(x) / \psi(x)$")
ax.set_xlabel('$x$')
ax.legend()
plt.show()
# -

# +
# Carry out the estimation with importance sampling

# Generate samples y distributed according to psi
u = np.random.rand(n)
x = np.random.randn(n)
y = (-1 + 2*(u > .5))*a + s*x

# Function of which we need to calculate the expectation wrt psi
h = lambda y: f(y)*pi(y)/psi(y)

# Check that y has the correct distribution
fig, ax = plt.subplots()
ax.hist(y, bins=30, density=True)
ax.plot(x_plot, h(x_plot))
plt.show()
hy = h(y)
# -

# + Calculate mean and variance of the estimator without and with IS
print(np.mean(fx), np.var(fx))
print(np.mean(hy), np.var(hy))
# -

# +
# Things can go wrong if the variance of the Gaussians in our Gaussian mixture
# are too small:
u = np.random.rand(n)
x = np.random.randn(n)

s = .1
g1, g2 = gaussian(-a, s), gaussian(a, s)
psi = lambda x: (1/2)*g1(x) + (1/2)*g2(x)
y = (-1 + 2*(u > .5))*a + s*x

h = lambda y: f(y)*pi(y)/psi(y)
hy = h(y)
print(np.mean(hy), np.var(hy))
# -

# # A more interesting example of importance sampling
# Assume that $\{Z_i\}_{i=1}^N$ are indepedent $\mathcal N(0, \sigma^2)$ random variables and
# define
# $$
# S_k = s_0 + \sum_{i=1}^k Z_k, \qquad k = 1, \dotsc, N.
# $$
# You may think of $S_k$ as, for example, the money left in the pocket of an
# poker player after $k$ games, and of $s_0 > 0$ as the money initially
# available to the player. If $s_0$ is sufficiently high relatively to $\sigma$,
# then the probability of ruin within the first $N$ games, given by
# $$
# I = \mathbb P \left(\min_{i = 1, \dotsc, N} (S_k) \leq 0 \right),
# $$
# is very small, so the relative accuracy of regular Monte Carlo simulation
# will not be very good.
#
# Now we will view $S := (S_1, \dotsc, S_N)$ as an $\mathbb R^N$-valued
# random variable. If we denote by $A$ the nonegative orthant $\mathbb R^N_{\geq 0}$,
# then clearly $ I = 1 - \mathbb E(I_{A}(S))$. The PDF of $S$ is given by
# $$
# \pi(s_1, \dotsc, s_N) = \frac{1}{\sqrt{2\pi\sigma^2}} \, \e^{-\frac{1}{2\sigma^2} \, ( (s_1 - s_0)^2 + (s_2 - s_1)^2 + \dotsb + (s_N - s_{N-1})^2 )}.
# $$
# In order to better estimate $I$, we will change the dynamics by adding a
# negative drift term. More precisely, we will use as our important
# distribution the PDF of the $\mathbb R^N$-valued random variable V obtained
# by
# $$
# V_k = s_0 - \sum_{i=1}^k b_i - \sum_{i=1}^k Z_i, \qquad k = 1, \dotsc, N,
# $$
# for deterministic $b_i$ (the drift) that we still need to choose.
# The associated PDF is given by:
# $$
# \psi(v_1, \dotsc, v_N) =  \frac{1}{\sqrt{2\pi\sigma^2}} \, \e^{-\frac{1}{2\sigma^2} \, ( (v_1 - v_0 + b_1)^2 + (v_2 - v_1 + b_2)^2 + \dotsb + (v_N - v_{N-1} + b_N)^2 )}., \qquad k = 1, \dotsc, N,
# $$
# The likelihood ratio can be calculated explicitly:
# $$
# g(x) = \frac{\pi(x)}{\psi(x)} = \exp \left( - \frac{1}{\sigma^2} \left( \sum_{i=1}^N b_i (x_i - x_{i-1}) - \frac{1}{2} \sum_{i=1}^N b_i^2 \right) \right).
# $$
# Now we can employ importance sampling.
# For simplicity, we will set $b_i = b$ (independent of $i$).

# +
s0, sigma, N, b = 1, .1, 10, -.1

# Target function to integrate
def f(x):
    return (np.min(x, axis=0) <= 0)*1.

# Likelihood ratio
def g(x):
    return np.exp(-(1/sigma**2)*(b*(x[-1] - s0) - b**2*N/2))

# -
# We will use regular Monte Carlo and importance sampling in order to
# approximate the probability of ruin. To avoid running out of RAM, but also
# just for the sake of illustration, we'll do updates of the mean and of the
# variance on the fly. These updates are based on the observation that, for
# numbers $\{x_i\}_{i\in \mathbb N}$ and with the notations
# $m_n = \frac{1}{n} \sum_{i=1}^n x_i$ and $q_n = \frac{1}{n} \sum_{i=1}^n x_i^2$,
# it holds that
# $$
# m_{n+1} = \frac{1}{n+1} (n \, m_n + x_{n+1}), \qquad q_{n+1} = \frac{1}{n+1} (n \, q_n + x_{n+1}^2).
# $$
# Note also that below we calculate the sample variance using an unbiased
# estimator, which is reflected by the presence of the factor $n/(n - 1)$.
# Without this, our estimator of the variance would be not unbiased but only
# asymptotically unbiased. However, since here we are interested more in the
# order of magnitude of the variance than in its precise value, both estimators
# are perfectly suitable. See [the wikipedia page on sample variance](https://en.wikipedia.org/wiki/Variance#Sample_variance) if you would like more information.

# +
n_per_slice, n_slices = 10**6, 10
n = n_per_slice * n_slices
mn, qn = 0, 0
mn_is, qn_is = 0, 0
for i in range(n_slices):
    # x = samples from the nominal distribution
    # y = samples from the importance distribution
    x = sigma * np.random.randn(N, n_per_slice)
    y = x + b
    x = s0 + np.cumsum(x, axis=0)
    y = s0 + np.cumsum(y, axis=0)
    fx, fy, gy = f(x), f(y), g(y)
    mn = 1/(i+1) * (mn*i + np.mean(fx))
    qn = 1/(i+1) * (qn*i + np.mean(fx**2))
    mn_is = 1/(i+1) * (mn_is*i + np.mean(fy*gy))
    qn_is = 1/(i+1) * (qn_is*i + np.mean((fy*gy)**2))

# Print the probability of ruin as calculated with each method
print(mn, mn_is)

# Calculate variance (times n) for both methods
var_no_reduction = n/(n - 1) * (qn - mn**2)
var_is = n/(n - 1) * (qn_is - mn_is**2)
print(var_no_reduction, var_is)
# Notice that, without variance reduction, the mean and variance calculated are
# very close. This is because the variance of a Bernoulli random variable of
# mean p is equal to p(1-p).

# -

# +
# Plot a few trajectories of the unmodified and modified dynamics. In other
# words, plot samples from pi and psi.
n_samples = 20
x0 = np.zeros(n_samples) + s0
trajectories = np.vstack((x0, x[:, :n_samples]))
trajectories_is = np.vstack((x0, y[:, :n_samples]))
fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(N + 1), trajectories, marker='.')
ax[1].plot(np.arange(N + 1), trajectories_is, marker='.')
ax[0].plot(np.arange(N + 1), np.zeros(N + 1), linestyle='--', color='k')
ax[1].plot(np.arange(N + 1), np.zeros(N + 1), linestyle='--', color='k')
ax[0].set_title("Samples from $\pi(\cdot)$")
ax[1].set_title("Samples from $\psi(\cdot)$")
plt.show()
