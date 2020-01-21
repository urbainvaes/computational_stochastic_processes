# Copyright (c) 2020 Urbain Vaes. All rights reserved.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Calculate pi using Monte Carlo methods

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##################
#  Minimal code  #
##################
n = int(1e5)

# Without fixing the seed
x, y = -1 + 2*np.random.rand(2, n)
in_circle = (x**2 + y**2 < 1)
print(4*sum(in_circle)/n)

# With deterministic seed
r = np.random.RandomState(0)
x, y = -1 + 2*r.rand(2, n)
in_circle = (x**2 + y**2 < 1)
print(4*sum(in_circle)/n)

####################################
#  Longer code with visualization  #
####################################

# Configure matplotlib (for plots)
matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(8, 8))

# Generate points on a circle
n_points = 200
thetas = np.linspace(0, 2*np.pi, n_points)
x = np.cos(thetas)
y = np.sin(thetas)

# Plot circle
fig, ax = plt.subplots()
ax.set_aspect(1)  # Ensure that the aspect ratio is correct
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.plot(x, y)

# Estimator of Monte Carlo method
# and the number of points used for its calculation
estimator, n_point = 0, 0

# Write value of the estimator in the plot
text = ax.text(.1, 1.02, r"$\hat I = {:.4f}$".format(estimator),
               fontsize=18,
               horizontalalignment='center',
               verticalalignment='center',
               transform=ax.transAxes)


# Monte Carlo method
def add_N_points(i):
    global estimator

    # Number of points added at each iteration
    N = 100

    x, y = -1 + 2*np.random.rand(2, N)  # Uniform distribution over [-1, 1]
    in_circle = (x**2 + y**2 < 1)

    # Evaluate indicator function (x 4) of the circle at (x, y)
    f_xy = (1/N) * sum(4 if b else 0 for b in in_circle)
    estimator = f_xy if i == 0 else (f_xy + (i - 1) * estimator)/i
    text.set_text(r"$\hat I = {:.4f}$".format(estimator))

    # Plot new points
    in_indices, out_indices = np.where(in_circle), np.where(np.invert(in_circle))
    x_in, y_in = x[in_indices], y[in_indices]
    x_out, y_out = x[out_indices], y[out_indices]
    ax.plot(x_in, y_in, marker='.', linestyle='',
            markersize=10, color='green')
    ax.plot(x_out, y_out, marker='.', linestyle='',
            markersize=10, color='red')

    print(estimator)


# Nothing to do for the initialization
def do_nothing():
    pass

# FuncAnimation will be called with range(n)
n = int(100)

# Create animation
anim = animation.FuncAnimation(fig, add_N_points, n,
                               init_func=do_nothing, repeat=False)

writer = animation.writers['ffmpeg'](fps=2, codec='libvpx-vp9', bitrate=3000)

# Save video ...
# anim.save('particles.webm', writer=writer)

# ... or show results on the fly (comment one)
plt.show()
