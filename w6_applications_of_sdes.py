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
matplotlib.rc('figure', figsize=(13, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=12)
matplotlib.rc('figure.subplot', hspace=.1)
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

def initial_condition(x):
    return (x >= -1)*(x <= 1)

def exact_solution(t, x):
    cdf = scipy.stats.norm.cdf
    return cdf((1 + x) / np.sqrt(2*t)) - cdf((-1 + x) / np.sqrt(2*t))

# Vector of times at which we will plot the exact and Monte Carlo solutions
T, N = 0, 100
t = np.linspace(0, 1, N + 1)
Δt = t[1] - t[0]

# Approximation by a Monte Carlo method

# We'll approximate the solution at a small number of space points
L, n, n_mc = 3, 400, 20
x_mc = np.linspace(-L, L, n_mc)

# We will use the same Brownian motions in our Monte Carlo estimator at each
# space point, but this is not necessary.
M = 10**3
ws = np.vstack((np.zeros(M), np.random.randn(N, M)))
ws = np.sqrt(Δt) * np.cumsum(ws, axis=0)

# Array to store the results of the MC estimation
mc_estimator = np.zeros((n_mc, N + 1))
for i, xi in enumerate(x_mc):
    mc_estimator[i] = np.mean(initial_condition(xi + np.sqrt(2)*ws), axis=1)

x = np.linspace(-L, L, n)
fig, ax = plt.subplots()
ax.set_title("Solving the heat equation by a Monte Carlo method")
fig.subplots_adjust(left=.05, bottom=.1, right=.98, top=.98)

# The variables employed in plot_time need to be defined globally
line, line_mc, text = None, None, None

# Function to plot the exact and approximate solutions at the i-th time step
def plot_time(i):
    global line, line_mc, text

    if i == 0:
        ax.clear()
        line, = ax.plot(x, initial_condition(x), label="Exact solution")
        line_mc, = ax.plot(x_mc, mc_estimator[:, i], linestyle='', marker='.',
                           label="Monte Carlo solution")
        ax.legend()
        ax.set_xlabel('$x$')
        text = ax.text(.1, .9, r"$t = {:.4f}$".format(0),
                       fontsize=18, horizontalalignment='center',
                       verticalalignment='center', transform=ax.transAxes)
    else:
        line.set_ydata(exact_solution(t[i], x))
        line_mc.set_ydata(mc_estimator[:, i])
        text.set_text(r"$t = {:.4f}$".format(t[i]))

def do_nothing():
    pass

# Create animation
anim = animation.FuncAnimation(fig, plot_time, list(range(N + 1)),
                               init_func=do_nothing, repeat=True)

# For the Jupyter notebook
plt.close(fig)
anim

# For python
# plt.show()
# -
