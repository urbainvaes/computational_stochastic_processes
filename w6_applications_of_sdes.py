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
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)
matplotlib.rc('animation', html='html5')
# -
# # The Feynman-Kac formula
# We illustrate below how the Feynman-Kac formula can be leveraged to solve the heat equataion
# $$
# \left\{
# \begin{aligned}
# & \partial_t u(t, x) = \Delta u(t, x), \qquad & t \in (0, T], ~ x \in \mathbb R, \\
# & u(0, x) = f(x), \qquad & x \in \mathbb R.
# \end{aligned}
# \right.
# $$
# This equation admits the exact solution
# $$
# u(t, x) = \frac{1}{\sqrt{4 \pi t}} \int_{-\infty}^{\infty} \exp \left( - \frac{(x-y)^2}{4t}\right) \, f(y) \, \mathrm d y,
# $$
# which is simply the convolution of the initial condition with the Green's function associated with the equation &ndash; the heat kernel.
# By the Feynman-Kac formula for autonomous equations, the solution admits the representation
# $$
# u(t, x) = \mathbb E(f(X_t)), \qquad X_t = x + \int_0^t \sqrt{2} \, \mathrm d W_t,
# $$
# where $W$ is a Brownian motion.
# Written more compactly,
# $$
# u(t, x) = \mathbb E(f(x + \sqrt{2} \, W_t)).
# $$
# We can therefore approximate $u(t, x)$ by Monte Carlo simulation:
# here we do not even need to employ a numerical method for SDEs,
# because we need only to simulate Brownian motion.
# In addition, since we can simulate Brownian motion exactly on discrete time points,
# our estimator of $u(t, x)$ will be unbiased.
#
# We will take the initial condition to be the indicator function of the interval $[-1, 1]$,
# i.e. $f(x) = I_{[-1, 1]}(x)$.
# For this choice, the exact solution can be expressed more explicitly
# $$
# u(t, x) = \frac{1}{\sqrt{4 \pi t}} \int_{-1}^{1} \exp \left( - \frac{(x-y)^2}{4t}\right) \, \mathrm d y
# =  \frac{1}{\sqrt{4 \pi t}} \int_{-1 + x}^{1 + x} \exp \left( - \frac{y^2}{4t}\right) \, \mathrm d y
# =  \frac{1}{\sqrt{2 \pi}} \int_{(-1 + x) / \sqrt{2t}}^{(1 + x) / \sqrt{2t}} \exp \left( - \frac{y^2}{2}\right) \, \mathrm d y,
# $$
# which can be calculated from the CDF a normally-distributed random variable.

# +
L, n = 3, 400

def exact_solution(t, x):
    cdf = scipy.stats.norm.cdf
    return cdf((1 + x) / np.sqrt(2*t)) - cdf((-1 + x) / np.sqrt(2*t))

x = np.linspace(-L, L, n)
fig, ax = plt.subplots()
def plot_time(t):
    ax.clear()
    ax.plot(x, exact_solution(t, x))

def do_nothing():
    pass

t = np.linspace(.01, 1, 100)
anim = animation.FuncAnimation(fig, plot_time, t,
                               init_func=do_nothing, repeat=False)
plt.close(fig)
anim
# -
# # Girsanov formula and importance sampling
# Let us consider the following SDE with constant diffusion coefficient and deterministic initial condition:
# $$
# \newcommand{\d}{\mathrm d}
# \newcommand{\expect}{\mathbb E}
# \d X_t = b(X_t) \, \d t + \sigma \, \d W_t, \qquad X_0 = x_0.
# $$
# The update formula of the Euler-Maruyama scheme for this equation is the following:
# $$
# X^{\Delta t}_{n+1} = X^{\Delta t}_n + b(X^{\Delta t}_n) \, \Delta t + \sigma \, \Delta W_n, \qquad X^{\Delta t}_0 = x_0, \qquad n = 0, \dotsc, N-1.
# $$
# This update formula defines a discrete-time stochastic process very similar to one we looked at a few weeks ago,
# when we calculated the probability of ruin of a gambler by importance sampling.
# Remember, in particular,
# that we derived an explicit expression for the PDF of $\{X^{\Delta t}_n\}_{n=1}^{N}$,
# viewed as a random variable in $\mathbb R^N$:
# the PDF is given by
# $$
# f_X^N(x_1, \dotsc, x_N) =
# \left|\frac{1}{\sqrt{2\pi\sigma^2\Delta t}}\right|^N \, \exp \left(-\frac{1}{2\sigma^2\Delta t} \sum_{k=0}^{N -1} \left|x_{k+1} - x_{k} - b(x_k) \Delta t \right|^2 \right).
# $$
# Let us now denote by $Y_t$ and $Y^{\Delta}_t$ the exact solution to the equation without drif tand its Euler-Maruyama approximation,
# i.e. with $b(\cdot) = 0$,
# and by $f_{Y}^N$ the associated PDF.
# A simple calculation shows the ratio of the densities is given by:
# $$
# M_N(x_1, \dotsc, x_N) := \frac{f_X^N(x_1, \dotsc, x_N)}{f_Y^N(x_1, \dotsc, x_N)} = \exp \left(\frac{1}{\sigma^2} \sum_{k=0}^{N -1} b(x_k) \, (x_{k+1} - x_{k}) - \frac{1}{2} |b(x_k)|^2 \Delta t \right).
# $$
# Since the right-hand side is strictly positive,
# we say in measure-theoretic terms that $f_X^N$ (or, to be more precise, the measure associated to it)
# is *absolutely continuous* with respect to $f_Y^N$.
# Since the reciprocal of the ratio is also positive, $f_Y^N$ is also absolutely continuous with respect to $f_X^N$,
# we say that the two measures are *equivalent*.
# This implies in particular that we can use $f_X^N$ to compute expectations with respect to $f_Y^N$, and vice versa:
# $$
# \expect_{Y^{\Delta t} \sim f_Y^N} [g(Y^{\Delta t})] = \expect_{X^{\Delta t} \sim f_X^N} [M_N(X^{\Delta t})^{-1} \, g(X^{\Delta t})]. \tag{1}
# $$
# where here $X^{\Delta t}$ and $Y^{\Delta t}$ are short notations for $(X^{\Delta t}_1, \dotsc, X^{\Delta t}_N)$ and $(Y^{\Delta t}_1, \dotsc, Y^{\Delta t}_N)$, respectively.
# Now observe, as we already did in the problem sheet,
# that if $X^{\Delta t}$ is obtained from the Euler-Maruyama scheme above,
# then we have
# $$
# M_N(X^{\Delta t}) = \exp \left(\frac{1}{\sigma^2} \sum_{k=0}^{N -1} \sigma \,b(X^{\Delta t}_k) \, \Delta W_k + \frac{1}{2} |b(X^{\Delta t}_k)|^2 \Delta t \right).
# $$
# Denoting by $\hat X^{\Delta t}$ the piecewise constant continuous-time interpolation of $X^{\Delta t}_k$,
# we can rewrite the previous equation as
# $$
# M_N(X^{\Delta t}) = \exp \left(\frac{1}{\sigma} \int_0^T \,b(\hat X^{\Delta t}_t) \, \d W_t + \frac{1}{2 \sigma^2} \int_0^t |b(\hat X^{\Delta t}_t)|^2 \d t \right).
# $$
# The Girsanov theorem shows that these considerations can be extended to the continuous-time processes $X_t$ (with drift $b$) and $Y_t$ (without drift).
# Roughly speaking, the theorem states that we can pass to the limit in equation $(1)$,
# in the sense that
# $$
# \expect [g(Y_t)] = \expect [M(X_t)^{-1} \, g(X_t)],
# $$
# where $M(t)$ admits the expression suggested by $(2)$:
# $$
# M(X_t) = \exp \left(\frac{1}{\sigma} \int_0^T \,b(X_t) \, \d W_t + \frac{1}{2 \sigma^2} \int_0^t |b(X_t)|^2 \d t \right).
# $$
# Note that, in this equation $W_t$ is the driving Brownian motion of the equation for $X_t$;
# in other words, it might be useful to see $W_t$ as a function of $X_t$ in this equation.
#
# Below we employ this result to estimate the probability that a Brownian motion exceeds a certain threshold:
# this is the continuous counterpart of the gambler's ruin.

# +
