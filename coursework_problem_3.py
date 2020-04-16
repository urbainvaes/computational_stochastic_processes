# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import numpy as np
import sympy as sym
import itertools
import matplotlib
import matplotlib.pyplot as plt
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', marker='')
matplotlib.rc('lines', markersize=10)
np.random.seed(0)
# -
# # Problem 3

# ## Question 1
# Using the definition of the Stratonovich integral,
# $$
# \newcommand{\e}{\mathrm e}
# \newcommand{\d}{\mathrm d}
# \newcommand{\expect}{\mathbb E}
# \newcommand{\var}{\mathrm{Var}}
# \int_0^t W_s \circ \d W_s = \lim_{N \to \infty} \frac{1}{2} \sum_{j=0}^{N-1} (W_{t^N_j} + W_{t^N_{j+1}})(W_{t^N_{j+1}} - W_{t^N_{j}}).
# $$
# We observe that the sum is a telescopic sum:
# $$
# \sum_{j=0}^{N-1} (W_{t^N_j} + W_{t^N_{j+1}})(W_{t^N_{j+1}} - W_{t^N_{j}}) = \sum_{j=0}^{N-1}|W_{t^N_{j+1}}|^2 - |W_{t^N_{j}}|^2 = |W_T|^2 - |W_0|^2 = |W_T|^2,
# $$
# which gives
# $$
# \int_0^T W_s \circ \d W_s = \frac{|W_T|^2}{2}.
# $$
# (This is of course valid for any value of $T \geq 0$.)

# Next, using the chain rule with $Y_t = f(W_t) = W_t^{m+1}/(m+1)$, we obtain
# $$
# \d Y_t = W_t^m \circ \d W_t.
# $$
# Integrating between $0$ and $T$, we deduce
# $$
# Y_t - Y_0 = \int_0^t W_s^m \circ \d W_s.
# $$
# Substituting the definition of $Y_t$, we conclude
# $$
# \int_0^t W_s^m \circ \d W_s = \frac{W_t^{m+1}}{m+1}.
# $$

# ## Question 2
# Applying the chain rule to $Y_t = \log(X_t)$,
# we obtain
# $$
# \d Y_t = \mu \, \d t + \sigma \, \d W_t.
# $$
# Integrating between $0$ and $t$, we deduce
# $$
# Y_t - Y_0 = \mu \, t + \sigma W_t.
# $$
# Substituting the definition of $Y_t$, we conclude
# $$
# X_t = X_0 \, \e^{\mu t + \sigma W_t}.
# $$

# ## Question 3
# Let $\mathbf X_t = (X_1(t), X_2(t))$, where $X_1(t) = (t - s)$ and $X_2(t) = (W_t - W_s)$.
# Clearly $\mathbf X_t$ is the solution to the following system of Stratonovich stochastic differential equations:
# $$
# \begin{aligned}
# dX_1(t) &= 1 \, \d t + 0 \circ \d W_t, \qquad    X_1(s) = 0,  \\
# dX_2(t) &= 0 \, \d t + 1 \circ \d W_t,   \qquad X_2(s) = 0.
# \end{aligned}
# $$
# Now let $Y_t = f(\mathbf X) = X_1(t) \, X_2(t)$, and notice that $Y_t = (t-s) \,(W_t - W_s) = (t-s) J^{s,t}_{(1)}$.
# By the chain rule
# $$
# \d Y_t = X_1(t) \, \d X_2(t) + X_2(t) \, \d X_1(t) = (t - s) \circ \d W_t + (W_t - W_s) \, \d t.
# $$
# (The expression in the middle here is only formal.)
# Integrating between $s$ and $t$, we deduce
# $$
# Y_t - Y_s = \int_s^t (t - s) \circ \d W_t + \int_s^t (W_t - W_s) \, \d t.
# $$
# The left-hand side is $(t - s) J^{s,t}_{(1)}$ and the right-hand side is $J^{s,t}_{(0, 1)} + J^{s,t}_{(1, 0)}$.

# Similarly, let $\mathbf X_t = (X_1(t), X_2(t))$, where $X_1(t) = (t - s)$ and $X_2(t) = J^{s,t}_{(1, 1)}$.
# By definition of $J^{s,t}_{(1, 1)}$,
# $\mathbf X_t$ is the solution to the following system of Stratonovich stochastic differential equations:
# $$
# \begin{aligned}
# dX_1(t) &= 1 \, \d t + 0 \circ \d W_t, \qquad    X_1(s) = 0,  \\
# dX_2(t) &= 0 \, \d t + J^{s,t}_{(1)} \circ \d W_t,   \qquad X_2(s) = 0.
# \end{aligned}
# $$
# Letting $Y_t = f(\mathbf X) = X_1(t) \, X_2(t)$ and applying the chain rule,
# we obtain
# $$
# \d Y_t = X_1(t) \, \d X_2(t) + X_2(t) \, \d X_1(t) = (t - s)  J^{s,t}_{(1)} \, \circ \d W_t + J^{s,t}_{(1,1)} \, \d t.
# $$
# (Again, the expression in the middle here is only formal.)
# Using the first identity, we obtain
# $$
# \d Y_t = (J^{s,t}_{(0, 1)} + J^{s,t}_{(1, 0)}) \, \circ \d W_t + J^{s,t}_{(1,1)} \, \d t.
# $$
# Integrating between $s$ and $t$, we deduce
# $$
# (t - s) \, J^{s,t}_{(1, 1)} = J^{s,t}_{(1, 0, 1)} + J^{s,t}_{(0, 1, 1)} + J^{s,t}_{(1, 1, 0)}.
# $$

# ## Question 4
# This derivation is very close to that of the Milstein scheme we carried out during the lecture.
# First, note that $f_{(0)}(x) = b(x)$ and $f_{(1)} = \sigma(x)$.
# Together with the fact that $\d X_t = b(X_t) \, \d t + \sigma(X_t) \, \d W_t$,
# this implies
# $$
# X_t = X_s + \int_s^t f_{(0)}(X_{u_1}) \, \d {u_1} + \int_s^t f_{(1)}(X_{u_1}) \circ \d W_{u_1}.
# \tag{1}
# $$
# Using the chain rule for $f_{(0)}$ and $f_{(1)}$, we deduce
# $$
# \begin{aligned}
# f_{(0)}(X_{u_1}) &= f_{(0)}(X_{s}) + \int_s^{u_1} b(X_{u_2}) \, f_{(0)}' (X_{u_2})  \, \d u_{2} + \int_s^{u_1} \sigma(X_{u_2}) \, f_{(0)}'(X_{u_2}) \circ \d W_{u_2}, \\
# f_{(1)}(X_{u_1}) &= f_{(1)}(X_{s}) + \int_s^{u_1} b(X_{u_2}) \, f_{(1)}' (X_{u_2})  \, \d u_{2} + \int_s^{u_1} \sigma(X_{u_2}) \, f_{(1)}'(X_{u_2}) \circ \d W_{u_2}.
# \end{aligned}
# $$
# This can be rewritten as
# $$
# \begin{aligned}
# f_{(0)}(X_{u_1}) &= f_{(0)}(X_{s}) + \int_s^{u_1} f_{(0, 0)} (X_{u_2})  \, \d u_{2} + \int_s^{u_1} f_{(1, 0)}(X_{u_2}) \circ \d W_{u_2}, \\
# f_{(1)}(X_{u_1}) &= f_{(1)}(X_{s}) + \int_s^{u_1} f_{(0, 1)} (X_{u_2})  \, \d u_{2} + \int_s^{u_1} f_{(1, 1)}(X_{u_2}) \circ \d W_{u_2}.
# \end{aligned}
# $$
# Employing the same reasoning, we can write
# $$
# \begin{aligned}
# f_{(0, 0)}(X_{u_2}) &= f_{(0, 0)}(X_{s}) + \dots \\
# f_{(1, 0)}(X_{u_2}) &= f_{(1, 0)}(X_{s}) + \dots \\
# f_{(0, 1)}(X_{u_2}) &= f_{(0, 1)}(X_{s}) + \dots \\
# f_{(1, 1)}(X_{u_2}) &= f_{(1, 1)}(X_{s}) + \dots,
# \end{aligned}
# $$
# where $\dots$ are (single) integral terms.
# Substituting in the expressions of $f_{(0)}(X_{u_1})$ and $f_{(1)}(X_{u_1})$, we obtain
# $$
# \begin{aligned}
# f_{(0)}(X_{u_1}) &= f_{(0)}(X_{s}) + f_{(0, 0)} (X_{s}) \, \int_s^{u_1}  \d u_{2} + f_{(1, 0)}(X_{s}) \, \int_s^{u_1} 1 \circ \d W_{u_2} + \dotsc, \\
# f_{(1)}(X_{u_1}) &= f_{(1)}(X_{s}) + f_{(0, 1)} (X_{s}) \, \int_s^{u_1}  \d u_{2} + f_{(1, 1)}(X_{s}) \, \int_s^{u_1} 1 \circ \d W_{u_2} + \dotsc.
# \end{aligned}
# $$
# Here $\dots$ are double integrals (remainder terms).
# Substituting these expression in equation $(1)$, we obtain
# $$
# \begin{aligned}
# X_t = \, &X_s + f_{(0)}(X_{s}) \, \int_s^t  \d {u_1} + f_{(1)}(X_{s}) \, \int_s^t \circ \d W_{u_1} \\
# &+ f_{(0, 0)} (X_{s}) \, \int_s^t  \int_s^{u_1} \d u_2 \, \d u_1 + f_{(0, 1)} (X_{s}) \int_s^t \int_s^{u_1}  \d u_{2} \circ \d W_{u_1} \\
# &+ f_{(1, 0)}(X_{s}) \, \int_s^t \int_s^{u_1} 1 \circ \d W_{u_2} \, \d {u_1} + f_{(1, 1)}(X_{s}) \int_s^t  \int_s^{u_1} 1 \circ \d W_{u_2}\circ \d W_{u_1} + \dots \\
# = \, & X_s + f_{(0)}(X_{s}) \, J^{s,t}_{(0)} + f_{(1)}(X_{s}) \, J^{s,t}_{(1)} \\
# &+ f_{(0, 0)} (X_{s}) \, J^{s,t}_{(0, 0)} + f_{(0, 1)} (X_{s}) \, J^{s,t}_{(0,1)} + f_{(1, 0)}(X_{s}) \, J^{s,t}_{(1, 0)} + f_{(1, 1)}(X_{s}) \, J^{s,t}_{(1, 1)} + \dots,
# \end{aligned}
# $$
# where here $\dots$ are triple integrals.

## Question 5
# The scheme is
# $$
# \begin{aligned}
# \frac{X_{n+1}^{\Delta t}}{X^{\Delta t}_n} = \,
# & 1 + \mu \, J^{t_n, t_{n+1}}_{(0)} + \sigma \, J^{t_n, t_{n+1}}_{(1)} \\
# & + \mu^2 \, J^{t_n, t_{n+1}}_{(0, 0)} + \mu \, \sigma (J^{t_n, t_{n+1}}_{(0, 1)} + J^{t_n, t_{n+1}}_{(1, 0)}) + \sigma^2 \, J^{t_n, t_{n+1}}_{(1, 1)} \\
# & + \mu \, \sigma^2 \, (J^{t_n, t_{n+1}}_{(0, 1, 1)} + J^{t_n, t_{n+1}}_{(1, 0, 1)} + J^{t_n, t_{n+1}}_{(1, 1, 0)}) + \sigma^3 \, J^{t_n, t_{n+1}}_{(1, 1, 1)} + \sigma^4 \, J^{t_n, t_{n+1}}_{(1, 1, 1, 1)}.
# \end{aligned}
# $$
# Using the identities derived in Question 3, we deduce
# $$
# \begin{aligned}
# \frac{X_{n+1}^{\Delta t}}{X^{\Delta t}_n} = \,
# & 1 + \mu \, J^{t_n, t_{n+1}}_{(0)} + \sigma \, J^{t_n, t_{n+1}}_{(1)} \\
# & + \mu^2 \, J^{t_n, t_{n+1}}_{(0, 0)} + \mu \, \sigma \, \Delta t \, J^{t_n, t_{n+1}}_{(1)} + \sigma^2 \, J^{t_n, t_{n+1}}_{(1, 1)} \\
# & + \mu \, \sigma^2 \, \Delta t \, J^{t_n, t_{n+1}}_{(1, 1)} + \sigma^3 \, J^{t_n, t_{n+1}}_{(1, 1, 1)} + \sigma^4 \, J^{t_n, t_{n+1}}_{(1, 1, 1, 1)}.
# \end{aligned}
# $$
# Using the notation $\Delta W_n = W_{n+1} - W_{n}$ and employing the result of Question 1,
# we obtain
# $$
# \begin{aligned}
# \frac{X_{n+1}^{\Delta t}}{X^{\Delta t}_n} = \,
# & 1 + \mu \, \Delta t + \sigma \, \Delta W_n \\
# & + \frac{1}{2} \, \mu^2 \, \Delta t^2 + \mu \, \sigma \, \Delta t \, \Delta W_n + \frac{1}{2} \sigma^2 \, \Delta W_n^2 \\
# & + \frac{1}{2} \, \mu \, \sigma^2 \, \Delta t \, \Delta W_n^2 + \frac{1}{6} \, \sigma^3 \, \Delta W_n^3 + \frac{1}{24} \, \sigma^4 \, \Delta W_n^4.
# \end{aligned}
# $$

## Question 6

# +
# Parameters
x0, μ, σ = 1, -1, 1

# Finest mesh
power_max, n_powers = 12, 8
N = 2**power_max
t = np.linspace(0, 1, N + 1)

# Number of replicas
M = 1000

# Brownian motion on finest mesh
Δt = t[1] - t[0]
Δw = np.sqrt(Δt) * np.random.randn(N, M)
w = np.vstack((np.zeros(M), np.cumsum(Δw, axis=0)))

# Exact solution
exact = x0*np.exp(μ*t + σ*w.T).T

# Second-order scheme
def stratonovich_integrator(t, w):
    x = np.zeros(w.shape)
    x[0] = x0
    dt, dw = t[1] - t[0], np.diff(w, axis=0)
    for j in range(len(t) - 1):
        factor = μ*dt + σ*dw[j] \
                + (1/2) * μ**2 * dt**2 + μ*σ * dt*dw[j] + (1/2) * σ**2 * dw[j]**2 \
                + (1/2) * μ*σ**2 * dt*dw[j]**2 + (1/6) * dw[j]**3 + (1/24) * dw[j]**4
        x[j+1] = x[j] + x[j] * factor
    return x

# Calculate the numerical solution for several time steps
xs = []
for i in range(n_powers):
    sol = stratonovich_integrator(t[::2**i], w[::2**i])
    xs.append(sol)

# Plot of some replicas
fig, ax = plt.subplots()
ax.plot(t, exact[:, 0], label=r"Exact solution")
for i, x_i in enumerate(xs):
    t_i, Δt_i = t[::2**i], Δt * 2**i
    ax.plot(t_i, x_i[:, 0],
            label=r"$\Delta t = 2^{{ {} }}$".format(np.log2(Δt_i)))
ax.legend()
ax.set_xlabel('$t$')
plt.show()
# -

# +
# Here we calculate the strong error at the final time only.
# This is not the only possibility
fig, ax = plt.subplots()
Δts = [Δt * 2**i for i in range(n_powers)]
errors = [np.mean(np.abs(x[-1] - exact[-1])) for x in xs]
ax.loglog(Δts, errors, label="Strong error", ls="", marker='.')
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)
ax.set_xlabel('$\Delta t$')
coeffs = np.polyfit(np.log2(Δts), np.log2(errors), 1)
ax.plot(Δts, 2**coeffs[1] * (Δts)**coeffs[0],
        label=r'${:.2f} \, \times \, \Delta t^{{ {:.2f} }}$'.
        format(2**coeffs[1], coeffs[0]))
ax.legend()
plt.show()
# -

# ## Question 7
# The scheme can be guessed by Taylor expansion.
# Noticing that the exact solution satisfies
# $$
# X_{(n+1)\Delta t} = X_{n\Delta t} \, \e^{\mu \Delta t + \sigma \Delta W_n},
# $$
# and Taylor-expanding the right-hand side, we obtain
# $$
# X_{(n+1)\Delta t} = X_{n\Delta} \left(1 + (\mu \Delta t + \sigma \Delta W_n) + \frac{1}{2}(\mu \Delta t + \sigma \Delta W_n)^2 + \dots \right).
# $$
# Including terms up to order 4 and keeping only the terms that scale as $\Delta t^2$ or less (using the fact that $\Delta W_n = \mathcal O(\Delta t^{1/2})$),
# the scheme is obtained.
