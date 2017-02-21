import numpy as np
from numba import jit

from matplotlib import pyplot as plt
from matplotlib import colors

image_counter = 60

# Burning Ship
# Julia Set

def burn_save_image(fig):
    global image_counter
    filename = "burning_%d.pdf" % image_counter
    image_counter += 1
    fig.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)


@jit
def ship(z, maxiter, horizon, log_horizon):
    c = z
    x = 0
    y = 0
    for n in range(maxiter):
        az = abs(z)
        if az > horizon:
            return n - np.log(np.log(az)) / np.log(2) + log_horizon
        # if abs(z) > 2:
        #    return n
        x, y = x * x - y * y - c.real, 2 * np.abs(x * y) - c.imag
        z = x + y * 1j
        # z = z*z*z*z + c
        # z = (np.abs(z.real) + np.abs(z.imag)*1j)**2 + c
        # z = z**3 + c
        # z = np.exp(z**3) + c
    return 0


@jit
def ship_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    horizon = 2.0 ** 40
    # horizon = 40
    log_horizon = np.log(np.log(horizon)) / np.log(2)
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = ship(r1[i] + 1j * r2[j], maxiter, horizon, log_horizon=0)
    return (r1, r2, n3)


# Create image function
def ship_image(xmin, xmax, ymin, ymax, width=8, height=11, maxiter=2048, cmap='RdGy', gamma=1):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = ship_set(xmin, xmax, ymin, ymax, img_width, img_height, maxiter)

    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ticks = np.arange(0, img_width, 3 * dpi)
    x_ticks = xmin + (xmax - xmin) * ticks / img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax - ymin) * ticks / img_width
    plt.yticks(ticks, y_ticks)
    # ax.set_title(cmap)
    ax.axis('off')
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    norm = colors.PowerNorm(gamma)
    ax.imshow(z.T, cmap=cmap, origin='lower', norm=norm)

    burn_save_image(fig)

    
#final images
ship_image(-2,2,-1,2,maxiter=2048, cmap='gist_ncar', gamma = 0.5)
ship_image(1.608,1.66,-0.02,0.06, maxiter = 2048, gamma = 0.5, cmap= 'gist_ncar')
ship_image(1.65,1.67,-0.002, 0.013, maxiter = 2048, gamma = 0.7, cmap= 'gist_ncar')