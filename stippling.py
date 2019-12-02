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

def weighted_voronoi_stippling(img, num_dots):
    def weighted_centroid(vertices, density):
        # raserize polygon -> get list of positions that are in the polygon
        # find weighted average (weighted by the grey level)
        pass

    max_iter = 100
    centroids = np.random.rand(num_dots, 2)
    for i in max_iter:
        # get voronoi cells
        new_cetroids = []
        vor = scipy.spatial.Voronoi(centroids)

        # get the vertices
        for region in vor.regions:
            vertices = vor.vertices[region + [region[0]], :]
            
            centroid = weighted_centroid(vertices, density)
            new_centroids.append(centroid)
        
        centroids = new_centroids

    dots = np.vstack(centroids)
    return dots

def dithering(img, style=None):
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

def dot(img, style='voronoi', num_dots=20000):
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

def show_dots(dot_coords, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    x = dot_coords[:, 1]
    y = dot_coords[:, 0]
    ax.scatter(x, -y, s=0.1)

def main():
    filename = './img/lenna.png'
    #filename = './img/david.jpg'

    rgb_img = util.load_img(filename)
    #rgb_img[:, :] = 0.5

    if np.max(rgb_img) > 1:
        rgb_img = rgb_img / 255
    grey_img = util.rgb2gray(rgb_img)

    dots_g = dot(grey_img, style='grid')
    dots_cg = dot(grey_img, style='cgrid')
    dots_dither = dot(grey_img, style='dithering')
    dots_odither = dot(grey_img, style='odithering')
    dots_rdither = dot(grey_img, style='rdithering')

    #fig = plt.figure("Input")

    #ax = fig.add_subplot(1,2,1)
    #ax.imshow(rgb_img)

    #ax = fig.add_subplot(1,2,2)
    #ax.imshow(grey_img, cmap='Greys_r')

    fig = plt.figure("Art")

    ax = fig.add_subplot(2,3,1)
    ax.set_aspect('equal')
    show_dots(dots_g, ax)

    ax = fig.add_subplot(2,3,2)
    ax.set_aspect('equal')
    show_dots(dots_cg, ax)
    
    ax = fig.add_subplot(2,3,3)
    ax.set_aspect('equal')
    show_dots(dots_dither, ax)

    ax = fig.add_subplot(2,3,4)
    ax.set_aspect('equal')
    show_dots(dots_rdither, ax)

    ax = fig.add_subplot(2,3,5)
    ax.set_aspect('equal')
    show_dots(dots_odither, ax)

    plt.show()

if __name__ == "__main__":
    main()
