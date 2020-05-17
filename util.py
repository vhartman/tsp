import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def load_img(filename):
    img = mpimg.imread(filename)

    return img

def distr_plts(n):
    if n == 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    elif n == 3:
        return 1, 3
    elif n == 4:
        return 2, 2
    elif n == 5:
        return 2, 3
    elif n == 6:
        return 2, 3
