# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(16, 11))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=12)
matplotlib.rc('figure.subplot', hspace=.3)
matplotlib.rc('figure.subplot', wspace=.1)
# -
# # Inferring the diffusion coefficient
# We illustrate the method seen in class for the simplest possible SDE with constant diffusion:
# $$
# \newcommand{\d}{\mathrm d}
# \d X_t = \sigma \, \d t, \qquad X_0 = 0.
# $$

# +
# True diffusion
σ = 1

# Final time
T = 1

# Finest grid points on which we will evaluate X
max_pow = 17
n = 2**max_pow
t = np.linspace(0, T, n + 1)

def estimator(N, x):
    # Check n is a multiple of N
    assert (n % N) == 0

    # Evaluate X on a grid of N points
    k = n // N
    x_N = x[np.arange(0, n + 1, k)]

    # Calculate quadratic variation based on grid
    qv = np.sum(np.diff(x_N)**2)

    # Return estimator
    return qv/T

Ns = 2**np.arange(max_pow + 1)

# Number of replicas
M = 100

estimators = np.zeros((M, len(Ns)))
for i in range(M):
    # Brownian increments and solution
    dw = np.sqrt(T/n) * np.random.randn(n)

    # Solution on the grid t
    x =  np.array([0, *np.cumsum(dw)])

    # Estimate the diffusion coefficient based on the realization x
    estimators[i] = [estimator(N, x) for N in Ns]

mean = np.mean(estimators, axis=0)
variance = np.var(estimators, axis=0)

fig, [ax1, ax2] = plt.subplots(2)
ax1.set_xlabel('$N$')
ax1.set_xscale('log', basex=2)
ax1.plot(Ns, mean, marker='.', label='$E [\hat \sigma^2_N]$')
ax1.plot(Ns, 0*Ns + σ**2, ls='--')
ax1.legend()
ax2.set_xlabel('$N$')
ax2.set_xscale('log', basex=2)
ax2.set_yscale('log', basey=2)
ax2.plot(Ns, variance, marker='.',
         label='$var[ \hat \sigma^2_N]$')
ax2.legend()
plt.show()
# -
# # Inferring the drift coefficient
# Here we infer the drift coefficient for the equation
# $$
# \d X_t = b \, \d t + \d W_t, \qquad X_0 = 0. \tag{1}
# $$
# In this case the density of the law $\mathbb P_X$ of $X ;= \{X_t\}_{t \in [0, T]}$ w.r.t. the law $\mathbb P_W$ of Brownian motion,
# both seen as measures on the space of continuous functions on $[0, T]$,
# is given by Girsanov's theorem:
# $$
# \frac{\d \mathbb P_X}{\d \mathbb P_Y} (X; b) = \exp \left(\int_0^T b \, \d X_t - \frac12 \int_0^T b^2 \, \d t \right)
#           = \exp \left(b \, X_T - \frac{1}{2} b^2 \, T \right).
# $$
# This density is called the *likelihood*.
# In measure-theoretic terms, the likelihood is the [Radon-Nikodym derivative](https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem) of
# $\mathbb P_X$ with respect to $\mathbb P_W$.
# When we say that $\d \mathbb P_X/\d \mathbb P_W$ is the density of $\mathbb P_X$ with respect to $\mathbb P_W$,
# what is meant is that, for any sufficiently well-behaved functional $g$ on $C([0, T])$,
# $$
#     \mathbb E_{X\sim\mathbb P_X} [g(X)] =  \int g \, \d \mathbb P_X
#     = \int g \, \frac{\d \mathbb P_X}{\d \mathbb P_W} \, \d \mathbb P_W
#     = \mathbb E_{W \sim \mathbb P_W} \left[g(W)  \frac{\d \mathbb P_X}{\d \mathbb P_W}(W; b)\right]
# $$
# You may wonder whether there is a reason for calculating the density w.r.t. the law of Browian motion rather than
# w.r.t. another probability measure on $C([D, T])$ that is also *equivalent* to the law of $X$?
#
# It turns out that the choice of the measure with respect to which we write the density of the law of $X$ does not matter:
# the maximum likelihood estimation would produce the same estimator if we used the density w.r.t. the law of $Y_t = W_t + t$,
# for example. Convince yourself of this!

# Calculating the density w.r.t. the law of Brownian motion is however convenient,
# because the density is given directly by Girsanov's theorem.
# We emphasize again there is no analogue of Lebesgue measure on an infinite-dimensional Banach spaces,
# which is the reason why we need to write the density w.r.t. another measure in the first place.
#
# The maximum likelihood estimator is the parameter $b$ that maximimizes the likelihood $M_T(X; b)$,
# where $X$ is a given observation.
# In our case, maximization produces the estimator
# $$
#     \hat b = X_T/T.
# $$
# This estimator happens to be unbiased, but this is usually not the case in more realistic situations.
# Note that, unlike inference for the diffusion coefficient,
# where we could obtain an estimation that is arbitrarily precise if we can have access to the solution at every $t \in [0, T]$,
# the only way to drive the variance of the maximum likelihood estimator to zero is to obtain more observations.

# Suppose that $X^{(1)}, \dotsc, X^{(J)}$ independently drawn from $\mathbb P_X$:
# by this, we mean that $X^{(1)}, \dotsc, X^{(J)}$ are i.i.d. random variables (living in a function space) whose law coincides with that of  the strong solution to $(1)$.
# The vector $(X^{(1)}, \dotsc, X^{(J)})$ defines a random variable on the space $C([0, T]) \times \dotsc \times C([0, T])$,
# whose density with respect to $\mathbb P_W \times \dotsc \times \mathbb P_W$ is given by
# $$
# M_T(X^{(1)}, \dotsc, X^{(J)}; b)
# = \exp \left(\int_0^T b \, \d X_t^{(1)} - \frac12 \int_0^T b^2 \, \d t \right) \times \dotsc \times \exp \left(\int_0^T b \, \d X_t^{(J)} - \frac12 \int_0^T b^2 \, \d t \right)
# $$
# Maximizing the likelihood w.r.t. $b$ produces the estimator
# $$
# \hat b = \frac{1}{J} \sum_{j=1}^J X^{(j)}_T.
# $$
# which is also unbiased and has variance $\mathrm{var}[X^{(0)}_T]/J$.
# -
