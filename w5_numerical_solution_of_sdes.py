# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=10)
matplotlib.rc('figure.subplot', hspace=.4)
# -
# # Taylor methods for stochastic differential equations

# +
def stochastic_integrator(t, x0_gen, drift, diff, m,
                          method="EM", diff_prime=None,
                          save_paths=True):
    """ Euler-Maruyama and Milstein schemes

    :t: Vector of times
    :x0_gen: Sampler for the initial condition
    :drift: Drift coefficient
    :diff: Diffusion coefficient
    :m: Number of replicas
    :method: Method to use (EM or Milstein)
    :diff_prime: Derivative of the diffusion coefficient (for Milstein)
    :save_paths: Return full paths or just final time
    """

    def step_em(x, dt):
        dw = np.sqrt(dt) * np.random.randn(len(x))
        return x + drift(x) * dt + diff(x) * dw, dw

    def step_milstein(x, dt):
        dw = np.sqrt(dt) * np.random.randn(len(x))
        extra_term = 1/2 * (diff_prime(x) * diff(x)) * (dw**2 - dt)
        return x + drift(x) * dt + diff(x) * dw + extra_term, dw

    step = step_em if method == "EM" else step_milstein
    n_times = len(t)

    if save_paths:

        # Matrix to store the solutions
        result = np.zeros((n_times, m))
        result[0] = x0_gen(m)

        # Matrix to store the Brownian motions (in order to
        # compute the exact solution - see below)
        w = np.zeros((n_times, m))

        for i in range(n_times - 1):
            result[i + 1], dw = step(result[i], t[i+1] - t[i])
            w[i + 1] = w[i] + dw

        return result, w

    # If we want only the solution at the final time
    result = x0_gen(m)
    w = np.zeros(m)

    for i in range(n_times - 1):
        result, dw = step(result, t[i+1] - t[i])
        w += dw

    return result, w

# Simulation parameters
T, n, m = 5, 100, 20

# Vector of times
t = np.linspace(0, T, n + 1)
# -
# We first illustrate the Euler--Maruyama for the Ornstein–Uhlenbeck process
# $$
# \newcommand{\d}{\mathrm d}
# \newcommand{\e}{\mathrm e}
# \newcommand{\cov}{\mathrm{cov}}
# \newcommand{\var}{\mathrm{var}}
# \d X_t = \theta (\mu -  X_t) \, \d t + \sigma \, \d W_t.
# $$

# +
# Parameters for the OU process
mu, theta, sigma = -1, 1, .2

# Drift and diffusion coefficients
drift = lambda x: theta*(mu - x)
diff = lambda x: sigma

# Initial condition, here ~ N(1, 1/25)
s = 1/5
x0 = lambda m: 1 + s*np.random.randn(m)

# Calculate solution by stochastic_integrator
x, _ = stochastic_integrator(t, x0, drift, diff, m)

# Plot of the solution
fig, ax = plt.subplots()
ax.set_title("Ornstein–Uhlenbeck process")
ax.set_xlabel("$t$")
ax.plot(t, x)
plt.show()
# -
# # Strong order of convergence
# Here we consider geometric Brownian motion, for which an
# exact solution is available:
# $$
# \d X_t = \mu \, X_t \, \d t + \sigma \, X_t \, \d t.
# $$

# +
# Parameters of the equation
mu, sigma = 1, .5

# Simulation parameters
T, n, m = 1, 100, 3

# Vector of times
t = np.linspace(0, T, n + 1)

# Drift and diffusion coefficients
drift = lambda x: mu*x
diff = lambda x: sigma*x
diff_prime = lambda x: sigma

# Initial condition, here deterministic
x0_deterministic = 1
x0 = lambda m: x0_deterministic * np.ones(m)

# Euler-Maruyama
x, w = stochastic_integrator(t, x0, drift, diff, m)

def exact_solution(t, w):
    # Make t a matrix, so that Python can do (t + w)
    t.shape = (len(t), 1)
    return x0_deterministic * np.exp((mu - sigma**2/2)*t + sigma * w)

# Plot of the solution
fig, ax = plt.subplots()
ax.set_xlabel("$t$")
ax.set_title("Geometric Brownian motion (dashed = exact)")
ax.plot(t, x)
# Reset color cycle
ax.set_prop_cycle(None)
ax.plot(t, exact_solution(t, w), linestyle="--")
plt.show()
# -
# We calculate the strong order of convergence by looking at the dependence on the time step $\Delta t$ of the strong error:
# $$
# \varepsilon = \mathbb E \left[ \sup_{n \Delta t \leq T} |X^{\Delta t}_n - X_{n\Delta t}| \right],
# $$
# where $X_t$ denotes the exact solution and $X^{\Delta t}_n$
# denotes the numerical solution at step $n$,
# where $\Delta t$ is used as the time step.
# An alternative would have been to look at the strong error
# at the final time:
# $$
# \varepsilon_T = \mathbb E |X^{\Delta t}_{T/\Delta t} - X_{T}|.
# $$

# +
def strong_error(x, x_exact):
    sup_interval = np.max(np.abs(x - x_exact), axis=0)
    return np.mean(sup_interval)


m, len_ns = 300, 10
ns = np.logspace(2, 4, len_ns)
ns = np.array([int(n) for n in ns])
strong_errors = np.zeros(len_ns)


def plot_errors(Δts, errors, error_type, method):
    # Fit to estimate order of convergence
    coeffs = np.polyfit(np.log2(Δts), np.log2(errors), 1)

    # Plot
    fig, ax = plt.subplots()
    ax.set_title("{} error of the {} scheme".format(error_type, method))
    if error_type == "strong":
        ylabel = r"$\sup \{ |X^{\Delta t}_n  - X_{n \Delta t}|:" \
                + "n \Delta t \in [0, T] \}$"
    elif error_type == "Weak":
        ylabel = r"$|E[f(X^{\Delta}_{T/\Delta}] - E[f(X_t)] |$"
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(ylabel)
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)
    ax.plot(Δts, errors, linestyle='', marker='.')
    ax.plot(Δts, 2**coeffs[1] * (Δts)**coeffs[0],
            label=r'${:.2f} \, \times \, \Delta t^{{ {:.2f} }}$'.
            format(2**coeffs[1], coeffs[0]))
    ax.legend()
    plt.show()

def calculate_strong_convergence(method):
    for i, n in enumerate(ns):
        t = np.linspace(0, T, n)
        x, w = stochastic_integrator(t, x0, drift, diff, m,
                                     method=method, diff_prime=diff_prime)
        x_exact = exact_solution(t, w)
        strong_errors[i] = strong_error(x, x_exact)
    plot_errors(T/ns, strong_errors, "strong", method)


calculate_strong_convergence("EM")
calculate_strong_convergence("Milstein")
# -
# # Weak order of convergence
# Here we consider again geometric Brownian motion, and we
# will estimate the weak error at the final time for the
# observable $f(x) = x$, i.e. we will estimate
# $$
# \varepsilon = | \mathbb E [X^{\Delta t}_{T/\Delta t}] - \mathbb E [X_{T}] |.
# $$
# The weak order of convergence is much more difficult to confirm numerically,
# because an accurate estimation requires that the Monte Carlo error on
# $\mathbb E [X^{\Delta t}]$ be small in comparison with the weak error.
# For this reason, we select below parameters such that the system is only a
# little noisy - the deterministic part of the dynamics dominates.
# Our choice of the parameters $\mu$ and $\sigma$ is taken from Des Higham's
# [introduction to the numerical solution of stochastic differential equations]
# (https://epubs.siam.org/doi/pdf/10.1137/S0036144500378302).
m, len_ns = 10**5, 5
ns = np.logspace(1, 3, len_ns)
ns = np.array([int(n) for n in ns])
weak_errors = np.zeros(len_ns)

# Simulation parameters
T, n, m = 1, 100, 3

# Vector of times
t = np.linspace(0, T, n + 1)

# Parameters
mu, sigma = 2, .1

# Drift and diffusion coefficients
drift = lambda x: mu*x
diff = lambda x: sigma*x
diff_prime = lambda x: sigma

def calculate_weak_convergence(method):
    for i, n in enumerate(ns):
        t = np.linspace(0, T, n)

        # Array to store the value of the numerical solution
        # at the last iteration
        result = []
        result, _ = stochastic_integrator(
                t, x0, drift, diff, m, method=method,
                diff_prime=diff_prime, save_paths=False)
        exact_expectation = np.exp(mu*T)
        weak_errors[i] = np.mean(np.abs(exact_expectation - np.mean(result)))
        print(weak_errors)
    plot_errors(T/ns, weak_errors, "Weak", method)

calculate_weak_convergence("EM")
