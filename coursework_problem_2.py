# +
# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import scipy.stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize
import scipy.integrate
# -

# +
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('figure', figsize=(13, 8))
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=12)
matplotlib.rc('figure.subplot', hspace=.1)
matplotlib.rc('animation', html='html5')
np.random.seed(0)
# -
# # Problem 2
# In this problem $X_t$ is the solution to the Ornstein-Uhlenbeck equation:
# \begin{align}
# \newcommand{\e}{\mathrm e}
# \newcommand{\d}{\mathrm d}
# \newcommand{\expect}{\mathbb E}
# \newcommand{\var}{\mathrm{Var}}
# \d X_t = - \theta (X_t - \mu) \, \d t + \sigma \, \d W_t.
# \end{align}

# ## Question 1
# Without the noise term, the solution would be $X_t = \mu + (X_0 - \mu) \, \e^{- \theta t}$.
# This suggests using Itô's formula with $Y_t = f(X_t, t) = \e^{\theta t}(X_t - \mu)$,
# which gives
# \begin{align}
# \d Y_t = \e^{\theta t} \, \theta \, (X_t - \mu) + \e^{\theta t} \, (- \theta \, (X_t - \mu) \, \d t + \sigma \, \d W_t)
#        = \e^{\theta t} \, \sigma \,  \d W_t.
# \end{align}
# This implies
# $$
# Y_t = Y_0 + \int_0^{t} \e^{\theta s} \, \sigma \, \d W_s = (X_0 - \mu) + \int_0^t \e^{\theta s} \, \sigma \, \d W_s,
# $$
# which gives
# $$
# X_t = \mu + (X_0 - \mu) \, \e^{-\theta \, t} + \sigma \, \int_0^t \e^{-\theta(t - s)} \, \d W_s.
# $$
# By Itô's isometry, the second moment at the final time is given by
# $$
# \begin{aligned}
# \expect |X_T|^2 &= |\expect[X_T]|^2 + \var[X_T] = |\mu + (\expect[X_0] - \mu) \, \e^{-\theta \, t}|^2 + \var[X_0] \, \e^{-2 \theta t} + \sigma^2 \, \int_0^T \e^{-2 \theta(T - s)} \, \d s \\
#                 &= \left|-1 + 2.5 \e^{-1} \right|^2 + \frac{1}{12} \, \e^{-2} + (1 - \e^{-2}) = 0.882...
# \end{aligned}
# $$

# ## Question 2
# Since
# $$
# I - I_N = \int_0^T (f(s) - f_N(s)) \, \d W_s,
# $$
# Itô's isometry implies
# $$
# \expect |I_N - I|^2 = \expect \left[ \int_0^T |f(s) - f_N(s)|^2 \, \d s \right].
# $$
# The right-hand side converges to zero as $N \to \infty$ because $f_N \to f$ in $L^2([0, T])$.
# Consequently, the left-hand side also converges to zero as $N \to \infty$,
# i.e. $I_N \to I$ in $L^2(\Omega)$ and thus also in distribution.

# On the other-hand,
# $$
# I_N = \sum_{k=0}^{N-1} f(k \Delta_N) \, (W_{(k+1) \Delta_N} - W_{k \Delta_N}).
# $$
# Since the right-hand side is a sum of independent normally distributed random variables,
# for any $N$ the integral $I_N$ is a normally distributed random variable.
# The mean of $I_N$ is 0 and the variance is
# $$
# \var[I_N] = \sum_{k=0}^{N-1} |f(k \Delta_N)|^2 \, \Delta_n = \int_0^T |f_N(s)|^2 \, \d s.
# $$
# Since $f_N \to f$ in $L^2([0, T])$ as $N \to \infty$,
# we deduce that $\var[I_N] \to \int_0^T |f(s)|^2 \, \d s$ as $N \to \infty$.
# Therefore $I_N \to \mathcal N\left(0, \int_0^T |f(s)|^2 \, \d s\right)$ in distribution as $N \to \infty$.

# By uniqueness of the limit in distribution, we conclude
# $$
# I \sim \mathcal N \left(0, \int_0^T |f(s)|^2 \, \d s\right).
# $$

# ## Question 3

# Generalizing slightly the result in Question 1, we obtain that
# $$
# X_t = \mu + (X_s - \mu) \, \e^{-\theta(t - s)} + \sigma \, \int_s^t \e^{-\theta(t - u)} \, \d W_u,
# $$
# for any two times $0 \leq s \leq t$.
# Letting $s = n \Delta t$ and $t = (n+1) \Delta t$,
# this gives
# $$
# X_{(n+1)\Delta t} = \mu + (X_{n\Delta t} - \mu) \, \e^{-\theta \Delta t} + \sigma \, \int_{n\Delta t}^{(n+1)\Delta t} \e^{-\theta((n+1)\Delta t - u)} \, \d W_u.
# $$
# Using the result in Question 2, we deduce that the Itô integral has variance
# $$
# \int_{n\Delta t}^{(n+1)\Delta t} \e^{-\theta((n+1)\Delta t - u)} \, \d W_u \sim \mathcal N \left(0, \int_{n\Delta t}^{(n+1)\Delta t} \e^{-2\theta((n+1)\Delta t - u)}\, \d u \right).
# $$
# The integral corresponding to the variance can be calculated explicitly:
# $$
# \int_{0}^{\Delta t} \e^{-2\theta(\Delta t - u)}\, \d u = \int_{0}^{\Delta t} \e^{-2\theta u}\, \d u = \frac{1}{2 \theta} \, (1 - \e^{-2 \theta \Delta t}).
# $$
# We have thus shown that the exact solution satisfies
# $$
# X_{(n+1) \, \Delta t} = \mu + a(\Delta t) \, (X_{n\Delta t} - \mu) + b(\Delta t) \, \xi \qquad \text{in law,}
# $$
# where $\xi \sim \mathcal N(0, 1)$, $a(\Delta t) = \e^{-2 \theta \Delta t}$ and $b(\Delta t) = \sqrt{\frac{\sigma^2}{2 \theta} \, (1 - \e^{-2 \theta \Delta t})}$.
# This suggests the numerical scheme
# $$
# X^{\Delta t}_{n+1} = \mu + a(\Delta t) \, (X^{\Delta t}_{n} - \mu) + b(\Delta t) \, \xi.
# $$
# Since successive iterates obey the same relationship as the exact solution,
# it is clear that the associated weak error is zero:
# for any $n$, the PDF of $X^{\Delta t}_n$ is tha same as that of $X_{n \Delta t}$,
# and similarly for all the finite-dimensional distributions.

# ## Question 4
# Since, in law,
# $$
# X_T = \mu + a(1) \, (X_0 - \mu) + b(1) \, \xi,
# $$
# we calculate
# $$
# \expect |X_T|^2 = \expect|\mu + a(1) \, (X_0 - \mu)|^2 + |b(1)|^2 = |\mu + a(1) \, (\expect[X_0] - \mu)|^2 + |a(1)|^2 \, \var[X_0] + |b(1)|^2.
# $$

# +
# Global, fixed parameters
μ, σ = -1, np.sqrt(2)

# True value of drift
true_θ = 1

def coefficients(θ, Δt):
    a = np.exp(-θ*Δt)
    b = np.sqrt(σ**2/2/θ * (1 - np.exp(-2*θ*Δt)))
    return a, b

def generate_trajectory(N, M, Δt):

    # Noise
    ξ = np.random.randn(N, M)

    # Solution
    x = np.zeros((N + 1, M))
    x0 = 1 + np.random.rand(M)
    x[0] = x0

    # Coefficients of the numerical method
    a, b = coefficients(true_θ, Δt)

    # Time-stepping
    for j in range(N):
        x[j + 1] = μ + a * (x[j] - μ) + b * ξ[j]

    return x

# Exact second moment
a, b = coefficients(true_θ, 1)
mean_x0, var_x0 = 1.5, 1/12
exact_second_moment = (μ + a*(mean_x0 - μ))**2 + a**2*var_x0 + b**2.
print("Exact value: {}".format(exact_second_moment))

N, M, Δt = 100, 10**5, .01
x = generate_trajectory(N, M, Δt)
fx = x[-1]**2

# Here, the variance can be calculated either explicitly based on the exact
# solution, or approximately based on the data.
σf = np.sqrt(np.var(fx))

# Estimator and confidence interval
estimator = np.mean(fx)

left_99 = estimator + scipy.stats.norm.ppf(.005) * σf/np.sqrt(M)
right_99 = estimator + scipy.stats.norm.ppf(.995) * σf/np.sqrt(M)
print("99% confidence interval: [{}, {}]".format(left_99, right_99))
# -

# +
fix, ax = plt.subplots()
t = np.linspace(0, N*Δt, N + 1)
ax.plot(t, x[:, :20])
ax.set_xlabel('$t$')
plt.show()
# -
# ## Question 5
# Given $X^{\Delta t}_n$, the conditional distribution of $X^{\Delta t}_{n+1}$
# is Gaussian with mean $\mu + a(\Delta t) \, (X^{\Delta t}_n - \mu)$ and variance $|b(\Delta t)|^2$.
# Therefore
# $$
# f_{\hat X} (x_{0}, \dotsc, x_{N} \, | \, \vartheta) = I_{[1, 2]}(x_{0}) \, \left(\frac{1}{\sqrt{2 \pi \, |b(\Delta t)|^2}}\right)^N \, \prod_{k=0}^{N-1} \exp \left(- \frac{\left(\mu + a(\Delta t) \, (x_k - \mu) - x_{k+1}\right)^2}{2 \, |b(\Delta t)|^2}\right),
# $$
# where in fact, on the right-hand side $a(\Delta t) = a(\Delta t; \vartheta)$ and $b(\Delta t) = b(\Delta t; \vartheta)$,
# but we do not write this explicitly, in order to keep the notations concise.

# ## Questions 6 and 7
# For Question 7, the joint probability distribution function is given by
# $$
# f_{\theta, \hat X} (\vartheta, x_{0}, \dotsc, x_{N}) = g_{2,1}(\vartheta) \, I_{[1, 2]}(x_{0}) \, \left(\frac{1}{\sqrt{2 \pi \, |b(\Delta t)|^2}}\right)^N \, \prod_{k=0}^{N-1} \exp \left(- \frac{\left(\mu + a(\Delta t) \, (x_k - \mu) - x_{k+1}\right)^2}{2 \, |b(\Delta t)|^2}\right),
# $$
# where $g_{2,1}$ is the PDF of $\mathcal N(2, 1)$.
# The conditional probability distribution $f_{\theta, \hat X}$ is given by
# $$
# f_{\theta|\hat X}(\vartheta|x_0, \dotsc, x_N) = \frac{f_{\theta, \hat X} (\vartheta, x_{0}, \dotsc, x_{N})}{\int_{\mathbb R} f_{\theta, \hat X} (\vartheta, x_{0}, \dotsc, x_{N}) \, \d \vartheta}.
# $$

# +
# Log-likelihood up to a constant
def log_likelihood(x, θ, include_prior):
    # Number of terms in the sum
    N = len(x) - 1

    # Coefficients of the numerical method
    a, b = coefficients(θ, Δt)

    # Calculation of the log-likelihood
    logL = - N*np.log(b) - np.sum((μ + a*(x[:-1] - μ) - x[1:])**2)/2/b**2

    # Contribution of the prior
    prior_contribution = - (θ - 2)**2/2

    return logL + include_prior*prior_contribution

# Inference
N, M, Δt = 10**6, 1, .1
t = np.linspace(0, N*Δt, N + 1)
x = generate_trajectory(N, M, Δt)[:,0]

# Range where we suspect the estimators will be
θmin, θmax = .97, 1.03

# Useful functions
fmin = scipy.optimize.fminbound
finterp = scipy.interpolate.interp1d
fsolve = scipy.integrate.solve_ivp
fzeros = scipy.optimize.brentq

# ML and MAP estimators
θmle = fmin(lambda θ: - log_likelihood(x, θ, False), θmin, θmax)
θmap = fmin(lambda θ: - log_likelihood(x, θ, True), θmin, θmax)
print(θmle, θmap)
# -
# ## Question 8

# +
# Approximation of the PDF for θ conditional on the data. We calcuate the
# log-likelihood over 200 points in the interval and interpolate between.
n = 200
θs = np.linspace(θmin, θmax, n)
Lθ = np.zeros(n)

for i, θ in enumerate(θs):
    Lθ[i] = log_likelihood(x, θ, True)

# To bring the likelihood to values of order one, we subtract the value of
# log-likelihood at the MAP.
pdf = np.exp(Lθ - log_likelihood(x, θmap, True))
pdf = finterp(θs, pdf, kind='linear', fill_value='extrapolate')

# Calculate the CDF based on the PDF
cdf = fsolve(fun=lambda θ, _: pdf(θ), t_span=[θmin, θmax],
             y0=[0], t_eval=θs, atol=1e-15, rtol=1e-13).y[0]
cdf = finterp(θs, cdf/cdf[-1], kind='linear', fill_value='extrapolate')

# Calculate 99% confidence interval
left_99 = fzeros(lambda x: .005 - cdf(x), θmin, θmax)
right_99 = fzeros(lambda x: .995 - cdf(x), θmin, θmax)

# Plot the estimators
fig, ax = plt.subplots()
height = pdf(θmap) * 1.1
ax.vlines([left_99, right_99], ymin=0, ymax=height,
          color='r', linestyle='--', label='99% conf. int.')
ax.vlines([θmap], ymin=0, ymax=height, color='k', label='MAP')
ax.vlines([θmle], ymin=0, ymax=height, color='g', label='MLE')
ax.set_xlabel(r"$\theta$")
ax.set_xlim(θmin, θmax)
ax.set_ylim(0, height)
ax.plot(θs, pdf(θs), label='PDF (up to normalization)')
ax.plot(θs, cdf(θs), label='CDF')
ax.legend()
plt.show()
# -
