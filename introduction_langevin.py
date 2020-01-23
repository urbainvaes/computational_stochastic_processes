# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Draw solutions of the Langevin equation
# To generate videos, use:
#     python codes/introduction_langevin.py -g .1 -m 10 -n 600 -dt .005
#     python codes/introduction_langevin.py -g 10 -m 10 -n 600 -dt .001
#     python codes/introduction_langevin.py -g 1 -m 10 -n 600 -dt .003

import argparse
import numpy as np
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure matplotlib (for plots)
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=False)
matplotlib.rc('figure', figsize=(14, 8))

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--theta', type=float)
parser.add_argument('-b', '--beta', type=float)
parser.add_argument('-g', '--gamma', type=float)
parser.add_argument('-dt', '--dt', type=float)
parser.add_argument('-n', '--niter', type=int)
parser.add_argument('-m', '--nensemble', type=int)
parser.add_argument('-i', '--interactive', action='store_true')
args = parser.parse_args()

# Parameters of the problem
gamma = args.gamma if args.gamma else 1  # Friction
beta = args.beta if args.beta else 1  # Inverse temperature

# Parameters of the simulation
n = args.niter if args.niter else int(100)  # Number of iterations
m = args.nensemble if args.nensemble else int(10)  # Number of particles
dt = args.dt if args.dt else 0.0005  # Time step of the Euler-Maruyama method

# Define external potential and calculate its derivative
x = sym.symbols('x')
Vx = (1/2)*(1 - sym.cos(x))
λVx = sym.lambdify(x, Vx)
dVx = sym.lambdify(x, Vx.diff(x))

# Define initial condition (all particles start at (1, 1.5))
X0 = np.zeros((2, m))
X0[0] = X0[0] + 1
X0[1] = X0[1] + 1.5

# t = time, X = solution
t = 0
X = X0


# Define drift and diffusion coefficients
def drift(X):
    return np.vstack((X[1], -dVx(X[0]) - gamma*X[1]))


diffusion = np.sqrt(2*gamma/beta)


# Implement Euler-Maruyama step
def step(X, dt):
    dW = np.random.normal(0, np.sqrt(dt), m)
    X = X + drift(X) * dt
    X[1] += diffusion * dW  # Brownian motion acts only on p
    return X


# Calculate the values of the Hamiltonian on a grid
grid = np.linspace(- 1, 1, 200)
x_grid, y_grid = np.pi * grid, 4 * grid
x, y = np.meshgrid(x_grid, y_grid)
H = λVx(x) + 1/2 * y**2

# Create figure and set colormap
fig, ax = plt.subplots()
plt.set_cmap('GnBu')


def update(i):

    global X
    print(i)

    # Clear drawings of the previous iteration
    ax.clear()

    # Plot the contours of the Hamiltonian
    ax.contourf(x, y, H, levels=20)

    # Perform Euler-Maruyama steps
    steps_per_update = 5
    for j in range(steps_per_update):
        X = step(X, dt)

    # Move all the particles in same period
    X[0] = (X[0] + np.pi) % (2*np.pi) - np.pi

    # Plot the particles
    for j in range(m):
        ax.plot(X[0][j], X[1][j], marker='.', linestyle='', markersize=15)

    # Plot arrows for the drift
    drift_coefficient = drift(X)
    scaling = np.max(np.linalg.norm(drift_coefficient, axis=0))
    drift_rescaled = drift_coefficient / scaling
    ax.quiver(X[0], X[1], drift_rescaled[0], drift_rescaled[1],
              scale_units='xy', scale=2, width=0.002, headwidth=3.)

    # Configure axes
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-4, 4)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_xlabel("$q$")
    ax.set_ylabel("$p$")
    ax.text(.06, .05, "$t = {:.4f}$".format(i*dt), fontsize=18,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

if args.interactive:
    plt.ion()
    for i in range(n):
        update(i)
        plt.draw()
        plt.pause(.1)

else:
    anim = animation.FuncAnimation(fig, update, args.niter, repeat=False)
    writer = animation.writers['ffmpeg'](fps=8, codec='libvpx-vp9', bitrate=3000)
    anim.save('particles.webm', writer=writer)
