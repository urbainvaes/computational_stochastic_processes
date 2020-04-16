# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import time
import sympy as sym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(13, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=12)
matplotlib.rc('figure.subplot', hspace=.1)
# -
# # Problem 1
# The exact solution to the equation is (see lecture notes)
# $$
# \newcommand{\e}{\mathrm e}
# \newcommand{\d}{\mathrm d}
# \newcommand{\expect}{\mathbb E}
# \newcommand{\var}{\mathrm{Var}}
# \newcommand{\re}{\mathrm{Re}}
# X_t = \exp \left( \left(\mu - \frac{\sigma^2}{2}\right) t + \sigma \, W_t \right).
# $$

# ## Question 1
# Using the expression of the exact solution,
# we calculate
# $$
# \begin{aligned}
# \expect [X_t \bar X_t]
# &= \expect \left[ \exp \left( \left(\mu + \bar \mu - \frac{\sigma^2}{2} - \frac{\bar \sigma^2}{2}\right) t + (\sigma + \bar \sigma) \, W_t \right) \right] \\
# &= \e^{\left(2\re(\mu) - \re(\sigma^2)\right) t} \, \expect \left[ \e^{2\re(\sigma) \, W_t} \right]
# = \e^{\left(2\re(\mu) - \re(\sigma^2)\right) t} \, \e^{2|\re(\sigma)|^2 \, t}.
# \end{aligned}
# $$
# Writing $\sigma = a + bi$, notice that $\re(\sigma^2) = a^2 - b^2$ and $|\re(\sigma)|^2 = a^2$.
# We deduce
# $$
# \expect [X_t \bar X_t] = \e^{2\re(\mu)t - (a^2 + b^2)t} = \e^{(2 \re(\mu) - |\sigma|^2)t},
# $$
# which converges to $0$ as $t\to \infty$ if and only if $2 \re(\mu) + |\sigma|^2 < 0$.

# ## Question 2
# In the case of geometric Brownian motion, the $\theta$ Milstein scheme reads
# $$
# X^{\Delta t}_{n+1} =  X^{\Delta t}_n + \big( \theta \, \mu \, X^{\Delta t}_{n+1} + (1-\theta) \, \mu \, X^{\Delta t}_n \big) \, \Delta t +
# \sigma \, X^{\Delta t}_n \, \Delta W_n + \frac{1}{2} \sigma^2 \, X^{\Delta t}_n \big((\Delta W_n)^2 - \Delta t \big).
# $$
# Rearranging the terms, this gives
# $$
#     X^{\Delta t}_{n+1} =  \frac{1 + (1-\theta) \, \mu \, \Delta t + \sigma \, \Delta W_n + \frac{1}{2} \sigma^2 \, \big((\Delta W_n)^2 - \Delta t \big)}{1 - \mu \, \Delta t \, \theta} \, X^{\Delta t}_n.
# $$

# Then, with the usual reasoning,
# $$
# \begin{aligned}
#     R(\Delta t, \mu, \sigma, \theta)
# &= \expect \left|\frac{1 + (1-\theta) \, \mu \, \Delta t + \sigma \, \Delta W_n + \frac{1}{2} \sigma^2 \, \big((\Delta W_n)^2 - \Delta t \big)}{1 - \mu \, \Delta t \, \theta} \right|^2 \\
# &= \frac{|1 + (1-\theta) \, \mu \, \Delta t|^2 + \expect \left[ \left(\sigma \, \Delta W_n + \frac{1}{2} \sigma^2 \, \big((\Delta W_n)^2 - \Delta t \big) \right)\left(\bar \sigma \, \Delta W_n + \frac{1}{2} \bar \sigma^2 \, \big((\Delta W_n)^2 - \Delta t \big) \right) \right]}{|1 - \mu \, \Delta t \, \theta|^2} \\
# &= \frac{|1 + (1-\theta) \, \mu \, \Delta t|^2 + |\sigma|^2 \, \Delta t + \frac{1}{2} |\sigma|^4 \, \Delta t^2}{|1 - \mu \, \Delta t \, \theta|^2},
# \end{aligned}
# $$
# where we used that $\expect[\Delta W_n^4] = 3 \Delta t^2$ and $\expect[\Delta W_n] = \expect [\Delta W_n^3] = 0$.

# ## Question 3
# In the case where $\mu$ and $\sigma$ are real,
# the expression of $R$ simplifies to
# $$
# \begin{aligned}
#     R(\Delta t, \mu, \sigma, \theta)
# &= \frac{(1 + (1-\theta) \, \mu \, \Delta t)^2 + \sigma^2 \, \Delta t + \frac{1}{2} \sigma^4 \, \Delta t^2}{(1 - \mu \, \Delta t \, \theta)^2}.
# \end{aligned}
# $$
# The scheme is mean-square stable for geometric Brownian motion if and only if $R(\Delta t, \mu, \sigma, \theta) < 1$
# or equivalently
# $$
# (1 + (1-\theta) \, \mu \, \Delta t)^2 + \sigma^2 \, \Delta t + \frac{1}{2} \sigma^4 \, \Delta t^2 - (1 - \mu \, \Delta t \, \theta)^2 < 0.
# $$
# Letting $x := \mu \, \Delta t$ and $y = \sigma^2 \, \Delta t$ and rearranging, this reads
# $$
# \frac{1}{2} y^2 + y + (1 + (1-\theta) \, x)^2 - (1 - x \, \theta)^2 < 0
# \Leftrightarrow \frac{1}{2} y^2 + y + 2(1-\theta)x + x^2 < 0.
# $$
# For fixed $\theta$ and fixed $x$, this is a quadratic polynomial in $y$ of the form $p(y) := a y^2 + b y + c$.
# Since $a > 0$ this polynomial tends to $+ \infty$ as $|y| \to \infty$.
# Let $\Delta$ denote the associated discriminant: $\Delta = b^2 - 4\, a \, c$.
# If $\Delta \leq 0$, $p(y) \geq 0$ for all $y$: there is no value of $y$ for which the scheme is stable.
# If $\Delta > 0$, $p(y) < 0$ between the two roots, given by
# $$
# y^{\pm} = \frac{-b \pm  \sqrt{\Delta}}{2a} = -1 \pm \sqrt{\Delta}.
# $$

# +
a = 1/2
b = 1
c = lambda θ, x: (1 + (1 - θ)*x)**2 - (1 - θ*x)**2
Δ = lambda θ, x: b**2 - 4*a*c(θ, x)

fig, ax = plt.subplots()
thetas = [0, .25, .5, .75, 1]
x = np.linspace(-2.5, 0, 200)  #
ax.plot(x, np.maximum(0, - 2*x), c='k')
for θ in np.flip(thetas):
    xi = x[np.where(Δ(θ, x) > 0)]
    yi = - (b - np.sqrt(Δ(θ, xi))) / (2*a)
    ax.plot(xi, yi, label=r"$\theta$-method with $\theta = {}$".format(θ))
    ax.fill_between(xi, 0*xi, yi, alpha=.2)
ax.set_title(r"Region of mean-square stability of the θ Milstein scheme")
ax.set_ylabel(r"$\sigma^2 \, \Delta t$")
ax.set_xlabel(r"$\mu \, \Delta t$")
ax.set_xlim(-2.5, 1)
ax.set_ylim(0, 2.5)
ax.legend()
plt.show()
# -
# ## Question 4
# Here we consider only $\theta \in [0, 1]$, for simplicity.
# Guided by the plots in the previous questions,
# we notice that region close to the black line,
# a line which corresponds to boundary of the mean-square stability region of the continuous solution,
# is not part of the region of mean-square stability of the scheme for the values of $\theta$ plotted.
# Let us take $x = -1$ and show that there always exists $y < 2$
# (this condition is necessary to guarantee that the continuous solution corresponding to the parameter choice is mean-square stable),
# such that the scheme is not mean-square stable.
# When $x = -1$, the scheme is mean-square stable if and only if
# $$
# \frac{1}{2} y^2 + y + \theta^2 - (1 + \theta)^2 < 0 \Leftrightarrow \frac{1}{2} y^2 + y - 1 - 2\theta < 0.
# $$
# Clearly, for $\theta \in [0, 1]$ there always exists $y > 0$ close to 2 such that the left-hand side is positive.
# Therefore, there is no choice of $\theta$ for which the scheme is A-stable.
# If we had not restricted our attention to the interval $\theta \in [0, 1]$,
# then we would have noticed that the scheme is mean-square stable for $\theta \geq 3/2$.

# ## Question 5
# The condition in this case reads
# $$
# \frac{1}{2} \Delta t^2 + \Delta t + (1 - .75 \, \Delta t)^2 - (1 + .25 \Delta t)^2 < 0 \Leftrightarrow \frac{1}{2} \Delta t^2 + \Delta t - 2 \Delta t + \frac{1}{2} \Delta t^2 < 0,
# $$
# which simplifies to $\Delta t^2 - \Delta t < 0$, i.e. $\Delta t \in (0, 1)$.

# +
# Parameters
mu, sigma, theta = -1, 1, .25

# θ-Milstein scheme
def step_theta_milstein(x, dt):
    ξ = np.random.randn(len(x))
    factor = (1 + (1 - theta)*mu*dt + sigma*ξ*np.sqrt(dt) + (1/2)*sigma**2 * (ξ**2 - 1) * dt) / (1 - theta*mu*dt)
    return x * factor

# Number of replicas
m = 100000

# Number of steps
n = 100

# Critical time step
dt = 1

# Initial condition
x = np.zeros(m) + 1
y = np.zeros(m) + 1

# Time-stepping
for i in range(n):
    x = step_theta_milstein(x, dt * 2)
    y = step_theta_milstein(y, dt / 2)

# Print results
print(np.mean(x**2), np.mean(y**2))
# -

# ## Question 6
# The scheme applied to the test equation gives
# $$
# X^{\Delta t}_{n+1} = 1 + \theta \, \mu \, X^{\Delta t}_{n+1} \, \Delta t + (1 - \theta) \, \mu \, X^{\Delta t}_{n} \, \Delta t + \sigma \, X^{\Delta t}_{n} \, \xi_n \sqrt{\Delta t}.
# $$
# Rearranging terms, we obtain
# $$
# X^{\Delta t}_{n+1} = \frac{1 + (1 - \theta) \, \mu \, \Delta t + \sigma \, \xi_n \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \, X^{\Delta t}_{n},
# $$
# which gives
# $$
# |X^{\Delta t}_{n}| = |X^{\Delta t}_0| \, \prod_{k=0}^{n-1} \left| \frac{1 + (1 - \theta) \, \mu \, \Delta t + \sigma \, \xi_n \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \right|.
# $$
# Therefore
# $$
# \log|X^{\Delta t}_{n}| = \log|X^{\Delta t}_0| + n \left( \frac{1}{n} \sum_{k=0}^{n-1} \log\left| \frac{1 + (1 - \theta) \, \mu \, \Delta t + \sigma \, \xi_n \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \right| \right).
# $$
# By the law of large numbers,
# it holds that $|X^{\Delta t}_n| \to 0$ almost surely if
# $$
# E := \expect \left[ \log\left| \frac{1 + (1 - \theta) \, \mu \, \Delta t + \sigma \, \xi_n \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \right| \right] < 0
# $$
# (See the lecture notes for more details).
# Since $\xi_n$ takes the values -1 and 1 with equal probability,
# $$
# \begin{aligned}
# E &= \frac{1}{2} \log\left| \frac{1 + (1 - \theta) \, \mu \, \Delta t + \sigma \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \right| +
# \frac{1}{2} \log\left| \frac{1 + (1 - \theta) \, \mu \, \Delta t - \sigma \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \right| \\
# &= \log\sqrt{\left| \frac{1 + (1 - \theta) \, \mu \, \Delta t + \sigma \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \,
# \times \frac{1 + (1 - \theta) \, \mu \, \Delta t - \sigma \, \sqrt{\Delta t}}{1 - \theta \, \mu \, \Delta t} \right|}.
# \end{aligned}
# $$
# Simplifying, we obtain
# $$
# E = \log\sqrt{\frac{|1 + (1 - \theta) \, \mu \, \Delta t|^2 - \sigma^2 \,\Delta t}{|1 - \theta \, \mu \, \Delta t|^2}}
# $$
# Therefore $E < 0$ if and only if
# $$
# \frac{|1 + (1 - \theta) \, \mu \, \Delta t|^2 - \sigma^2 \,\Delta t}{|1 - \theta \, \mu \, \Delta t|^2} < 1
# \Leftrightarrow (1 - 2 \theta) \mu^2 \Delta t^2 + 2 \mu \Delta t - \sigma^2 \Delta t < 0.
# $$
# When $\theta = 1/2$, this condition coincides with the condition of asymptotic stability of the continuous equation.
