import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import copy

import tsp

DISP = True

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def dot(img, style='voronoi', num_dots=20000):
    # do all the halftoning

    def grid(img, compensate=False):
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

                    r = np.random.rand(g, 2)
                    r[:, 0] = r[:, 0] * cell_size[0] + i*cell_size[0]
                    r[:, 1] = r[:, 1] * cell_size[1] + j*cell_size[1]

                    dots.append(r)

            return np.vstack(dots)

        num_cells_per_axis = 40
        discretized, s = compute_mean_per_cell(img, num_cells_per_axis)

        gamma = int(num_dots / num_cells_per_axis**2)
        dots = distribute_dots(discretized, s, gamma)
        
        return dots

    def weighted_voronoi_stippling(img):
        pass

    def dithering(img):
        d = copy.deepcopy(img)
        dots = []

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                old = d[i,j]
                new = round(old)

                d[i, j] = new

                quant_error = old - new
                if i < img.shape[0] - 1:
                    d[i + 1][j    ] = d[i + 1][j    ] + quant_error * 7 / 16
                if i > 0 and j < img.shape[1] - 1:
                    d[i - 1][j + 1] = d[i - 1][j + 1] + quant_error * 3 / 16
                if j < img.shape[1] - 1:
                    d[i    ][j + 1] = d[i    ][j + 1] + quant_error * 5 / 16
                if i < img.shape[0] - 1 and j < img.shape[1] - 1:
                    d[i + 1][j + 1] = d[i + 1][j + 1] + quant_error * 1 / 16

        for i in range(2, img.shape[0]-2, 4):
            for j in range(2, img.shape[1]-2, 4):
                m = np.mean(d[i-2:i+2, j-2:j+2])

                if m < 0.3:
                    dots.append([i, j])

        return np.vstack(dots)

    if style == 'voronoi':
        return weighted_voronoi_stippling(img)
    elif style == 'grid':
        return grid(img)
    elif style == 'cgrid':
        return grid(img, compensate=True)
    elif style == 'dithering':
        return dithering(img)

def postprocess(dots, style=''):
    if style == 'thickness':
        pass

    if style == 'smooth':
        pass

def show_tsp(dots, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    ax.plot(dots[:, 1], -dots[:, 0], linewidth=0.1)

def show_dots(dot_coords, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    x = dot_coords[:, 1]
    y = dot_coords[:, 0]
    ax.scatter(x, -y, s=0.1)

def load_img(filename):
    img = mpimg.imread(filename)

    return img

def handle_args():
    parser = argparse.ArgumentParser(
        description='Transforms a given image into a traveling salesman halftone version of it.')

    parser.add_argument('filename', type=str,
                    help='The filename to the image')

    args = parser.parse_args()
    return args

def main():
    args = handle_args()
    filename = args.filename

    rgb_img = load_img(filename)
    if np.max(rgb_img) > 1:
        rgb_img = rgb_img / 255
    grey_img = rgb2gray(rgb_img)

    dots_g = dot(grey_img, style='grid')
    dots_cg = dot(grey_img, style='cgrid')
    dots_dither = dot(grey_img, style='dithering')

    line = tsp.tsp(dots_cg)

    if DISP:
        fig = plt.figure("Input")

        ax = fig.add_subplot(1,2,1)
        ax.imshow(rgb_img)

        ax = fig.add_subplot(1,2,2)
        ax.imshow(grey_img, cmap='Greys_r')

        fig = plt.figure("Art")

        ax = fig.add_subplot(1,3,1)
        ax.set_aspect('equal')
        show_dots(dots_g, ax)

        ax = fig.add_subplot(1,3,2)
        ax.set_aspect('equal')
        show_dots(dots_cg, ax)
        
        ax = fig.add_subplot(1,3,3)
        ax.set_aspect('equal')
        show_dots(dots_dither, ax)

        fig = plt.figure("TSP")
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        show_tsp(line, ax)
        
        plt.show()

if __name__ == "__main__":
    main()
