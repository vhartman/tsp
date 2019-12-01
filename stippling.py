import numpy as np
import matplotlib.pyplot as plt
import copy

import util 

def grid(img, compensate=False, num_dots=20000):
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

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            old = d[y,x]
            new = min([max([0, round(old)]), 1])

            d[y, x] = new

            quant_error = old - new
            if x < img.shape[1] - 1:
                d[y    , x + 1] = d[y    , x + 1] + quant_error * 7 / 16
            if x > 0 and y < img.shape[0] - 1:
                d[y + 1, x - 1] = d[y + 1, x - 1] + quant_error * 3 / 16
            if y < img.shape[0] - 1:
                d[y + 1, x    ] = d[y + 1, x    ] + quant_error * 5 / 16
            if x < img.shape[1] - 1 and y < img.shape[0] - 1:
                d[y + 1, x + 1] = d[y + 1, x + 1] + quant_error * 1 / 16

    for i in range(2, img.shape[0]-2, 4):
        for j in range(2, img.shape[1]-2, 4):
            m = np.mean(d[i-2:i+2, j-2:j+2])

            if m < 0.5:
                dots.append([i, j])

    return np.vstack(dots)

def dot(img, style='voronoi', num_dots=20000):
    # do all the halftoning
    if style == 'voronoi':
        return weighted_voronoi_stippling(img)
    elif style == 'grid':
        return grid(img, compensate=False, num_dots=num_dots)
    elif style == 'cgrid':
        return grid(img, compensate=True, num_dots=num_dots)
    elif style == 'dithering':
        return dithering(img)

def show_dots(dot_coords, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    x = dot_coords[:, 1]
    y = dot_coords[:, 0]
    ax.scatter(x, -y, s=0.1)

def main():
    #filename = './img/lenna.png'
    filename = './img/david.jpg'

    rgb_img = util.load_img(filename)
    #rgb_img[:, :] = 0.5

    if np.max(rgb_img) > 1:
        rgb_img = rgb_img / 255
    grey_img = util.rgb2gray(rgb_img)

    dots_g = dot(grey_img, style='grid')
    dots_cg = dot(grey_img, style='cgrid')
    dots_dither = dot(grey_img, style='dithering')

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

    plt.show()

if __name__ == "__main__":
    main()
