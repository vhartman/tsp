import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DISP = False

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def dot(img, style='voronoi', num_dots=10000):
    dots = np.zeros((num_dots, 2))

    # do all the halftoning
    def grid(img):
        pass

    def contrast_grid(img):
        pass

    def weighted_voronoi_stippling(img):
        pass

    def ordered_dithering(img):
        pass

    if style == 'voronoi':
        return weighted_voronoi_stippling(img)
    elif style == 'grid':
        return grid(img)
    elif style == 'cgrid':
        return contrast_grid(img)
    elif style == 'dithering':
        return ordered_dithering(img)


def tsp(dots):
    # https://developers.google.com/optimization/routing/tsp
    # mlrose
    ordered_dots = dots
    return ordered_dots

def postprocess(dots, style=''):
    if style == 'thickness':
        pass

    if style == 'smooth':
        pass

def show_tsp(dots, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    ax.plot(dots)

def show_dots(dot_coords, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    ax.scatter(dot_coords.T)

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
    grey_img = rgb2gray(rgb_img)

    dots = dot(grey_img)
    line = tsp(dots)

    if DISP:
        fig = plt.figure("Input")

        ax = fig.add_subplot(1,2,1)
        ax.imshow(rgb_img)

        ax = fig.add_subplot(1,2,2)
        ax.imshow(grey_img, cmap='Greys_r')

        fig = plt.figure("Art")

        ax = fig.add_subplot(1,2,1)
        show_dots(dots, ax)

        ax = fig.add_subplot(1,2,2)
        show_tsp(line)
        
        plt.show()

if __name__ == "__main__":
    main()
