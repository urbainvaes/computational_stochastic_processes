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
