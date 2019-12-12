import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection

import util
import tsp
import stippling

DISP = True

def postprocess(dots, img, style='', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    if style == 'thickness':
        segs = []
        lw = []
        for i in range(dots.shape[0]):
            segs.append([(dots[i-1, 1], -dots[i-1, 0]), (dots[i, 1], -dots[i, 0])])
            
            x = (dots[i-1, 1] + dots[i, 1])/2
            y = (dots[i-1, 0] + dots[i, 0])/2
            l = (1 - img[int(y), int(x)])
            lw.append(0.1 + l*1.5)

        lc = LineCollection(segs, linewidths=lw)
        ax.add_collection(lc)
        ax.autoscale()

    if style == 'smooth':
        pass

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

    rgb_img = util.load_img(filename)
    if np.max(rgb_img) > 1:
        rgb_img = rgb_img / 255
    grey_img = util.rgb2gray(rgb_img)

    #dots_g = stippling.dot(grey_img, style='grid')
    #dots_v = stippling.dot(grey_img, style='voronoi', num_dots=10000)
    #dots_cg = stippling.dot(grey_img, style='cgrid')
    dots_dither = stippling.dot(grey_img, style='dithering')

    print('Starting TSP')
    line = tsp.tsp(dots_dither, style='rnn')
    print('Altering tour')
    line = tsp.alter_tour(line, max_len=100)

    if DISP:
        fig = plt.figure("Input")

        ax = fig.add_subplot(1,2,1)
        ax.imshow(rgb_img)

        ax = fig.add_subplot(1,2,2)
        ax.imshow(grey_img, cmap='Greys_r')

        fig = plt.figure("Art")

        ax = fig.add_subplot(1,3,1)
        ax.set_aspect('equal')
        #stippling.show_dots(dots_g, ax)

        ax = fig.add_subplot(1,3,2)
        ax.set_aspect('equal')
        #stippling.show_dots(dots_cg, ax)
        
        ax = fig.add_subplot(1,3,3)
        ax.set_aspect('equal')
        stippling.show_dots(dots_dither, ax)

        fig = plt.figure("TSP")
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        postprocess(line, grey_img, 'thickness', ax)
        
        plt.show()

if __name__ == "__main__":
    main()
