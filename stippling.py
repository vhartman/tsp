import numpy as np
import matplotlib.pyplot as plt
import copy

import util 
import voronoi
import scipy.ndimage

import argparse

def grid(img, compensate=False, num_dots=20000, num_cells_per_axis=40):
    def compute_mean_per_cell(img, cells_per_axis):
        n = np.zeros((cells_per_axis, cells_per_axis))

        x_scale = int(img.shape[0] / cells_per_axis)
        y_scale = int(img.shape[1] / cells_per_axis)
        for i in range(n.shape[0]):
            for j in range(n.shape[1]):
                n[i, j] = np.mean(img[i*x_scale:(i+1)*x_scale, j*y_scale:(j+1)*y_scale])

        return n, (x_scale, y_scale)

    def distribute_dots(mu, cell_size, dots_per_cell):
        dots = []
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                g = gamma - int((gamma + 1)*mu[i,j])
                if compensate:
                    g = int(1./3*g**2)

                g = np.max([0, g])

                r = np.random.rand(g, 2)
                r[:, 0] = r[:, 0] * cell_size[0] + i*cell_size[0]
                r[:, 1] = r[:, 1] * cell_size[1] + j*cell_size[1]

                dots.append(r)

        return np.vstack(dots)

    discretized, s = compute_mean_per_cell(img, num_cells_per_axis)

    gamma = int(np.ceil(num_dots / num_cells_per_axis**2) * 1/np.mean(discretized))
    dots = distribute_dots(discretized, s, gamma)

    # prune, so that we have the desired number of dots
    if len(dots) > num_dots:
        d = np.random.choice(np.arange(0, len(dots)), len(dots)-num_dots, replace=False)
        dots = np.delete(dots, d, axis=0)
    
    return dots

def weighted_voronoi_stippling(img, num_dots):
    def normalize(D):
        Vmin, Vmax = D.min(), D.max()
        if Vmax - Vmin > 1e-5:
            D = (D-Vmin)/(Vmax-Vmin)
        else:
            D = np.zeros_like(D)
        return D

    max_iter = 10

    density = 1 - img
    zoom = (num_dots * 500) / (density.shape[0]*density.shape[1])
    zoom = int(round(np.sqrt(zoom)))
    density = scipy.ndimage.zoom(density, zoom, order=0)

    density = np.minimum(density, 0.8)
    density = density**2
    density = normalize(density)
    density = density.T
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)

    def initialization(n, D):
        """
        Return n points distributed over [xmin, xmax] x [ymin, ymax]
        according to (normalized) density distribution.
        with xmin, xmax = 0, density.shape[1]
             ymin, ymax = 0, density.shape[0]
        The algorithm here is a simple rejection sampling.
        """

        samples = []
        while len(samples) < n:
            # X = np.random.randint(0, D.shape[1], 10*n)
            # Y = np.random.randint(0, D.shape[0], 10*n)
            X = np.random.uniform(0, D.shape[1], 10*n)
            Y = np.random.uniform(0, D.shape[0], 10*n)
            P = np.random.uniform(0, 1, 10*n)
            index = 0
            while index < len(X) and len(samples) < n:
                x, y = X[index], Y[index]
                x_, y_ = int(np.floor(x)), int(np.floor(y))
                if P[index] < D[y_, x_]:
                    samples.append([x, y])
                index += 1
        return np.array(samples)

    centroids = initialization(num_dots, density)

    for i in range(max_iter):
        # get voronoi cells
        _, centroids = voronoi.centroids(centroids, density, density_P, density_Q)

    dots = np.vstack(centroids) / zoom
    return dots

def dithering(img, style=None):
    def floyd_steinberg(img):
        d = copy.deepcopy(img)

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                old = d[y,x]
                new = min([max([0, round(old)]), 1])

                d[y, x] = new

                quant_error = old - new
                if x < img.shape[1] - 1:
                    d[y    , x + 1] = d[y    , x + 1] + quant_error * 7 / 16.
                if x > 0 and y < img.shape[0] - 1:
                    d[y + 1, x - 1] = d[y + 1, x - 1] + quant_error * 3 / 16.
                if y < img.shape[0] - 1:
                    d[y + 1, x    ] = d[y + 1, x    ] + quant_error * 5 / 16.
                if x < img.shape[1] - 1 and y < img.shape[0] - 1:
                    d[y + 1, x + 1] = d[y + 1, x + 1] + quant_error * 1 / 16.

        return d

    d = floyd_steinberg(img)

    plt.imshow(img)
    plt.figure()
    plt.imshow(d)
    plt.show()

    dots = []
    gs = 2
    for i in range(gs, img.shape[0]-gs, gs*2):
        for j in range(gs, img.shape[1]-gs, gs*2):
            if style is None:
                m = np.mean(d[i-gs:i+gs, j-gs:j+gs])
                if m < 0.5:
                    dots.append([i, j])
            elif style == 'rand':
                s = (gs*2)**2 - np.sum(d[i-gs:i+gs, j-gs:j+gs])
                s = s**1.5 / 10

                r = np.random.rand(int(s), 2)
                r[:, 0] = r[:, 0] * gs*2 + i
                r[:, 1] = r[:, 1] * gs*2 + j

                dots.append(r)

            elif style == 'ordered':
                g = (gs*2)**2 - np.sum(d[i-gs:i+gs, j-gs:j+gs])
                r = []
                if g == 16:
                    for k in range(gs*2):
                        for l in range(gs*2):
                            r.append([i, j])
                elif g >= 10:
                    for k in range(gs*2):
                        for l in range(gs*2):
                            if (k + l) % 2 == 0:
                                r.append([i, j])
                elif g >= 3:
                    r.append([0, 0])
                    r.append([2, 0])
                    r.append([0, 2])
                    r.append([2, 2])
                elif g > 1:
                    r.append([0, 0])
                    r.append([2, 2])
                elif g == 1:
                    r.append([0, 0])
                    r.append([2, 2])

                if len(r) > 0:
                    r = np.vstack(r)
                    dots.append(r)

    return np.vstack(dots)

def stipple(img, style='voronoi', num_dots=20000):
    # do all the halftoning
    if style == 'voronoi':
        return weighted_voronoi_stippling(img, num_dots)
    elif style == 'grid':
        return grid(img, compensate=False, num_dots=num_dots)
    elif style == 'cgrid':
        return grid(img, compensate=True, num_dots=num_dots)
    elif style == 'dithering':
        return dithering(img)
    elif style == 'odithering':
        return dithering(img, style='ordered')
    elif style == 'rdithering':
        return dithering(img, style='rand')

def show_dots(dot_coords, ax=None, color="black"):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    x = dot_coords[:, 1]
    y = dot_coords[:, 0]
    ax.scatter(x, -y, s=0.1, color=color)

    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
    x, y = util.distr_plts(len(methods))

    fig = plt.figure("Art")
    for i, method in enumerate(methods):
        dots = stipple(grey_img, style=method, num_dots=args.num_dots)

        ax = fig.add_subplot(x,y,i+1)
        ax.set_title(method)
        show_dots(dots, ax)

    plt.show()

if __name__ == "__main__":
    main()
