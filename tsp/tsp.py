import numpy as np
import matplotlib.pyplot as plt

import tsp.christofides as christofides
import tsp.greedy as greedy
import tsp.simulated_annealing as sa

import tsp.improve as imp

import tsp.util as util

# match twice and stitch
def mts(dots):
    # initial matching: minimum weight matching
    # second matching: minimum weight matching with previous edges cost set to inf
    # stitching
    return None

def optimize(dots, method='nn'):
    # https://developers.google.com/optimization/routing/tsp
    # mlrose
    print(method)

    if method == 'nn':
        return greedy.nn(dots)
    elif method == 'rnn':
        return greedy.repeated_nn(dots, 10)
    elif method == "nf":
        return greedy.nf(dots)
    elif method == 'sa':
        return sa.simulated_annealing(dots)
    elif method == 'christofides':
        return christofides.christofides(dots)

    return None

def improve(tour, method='2opt', **kwargs):
    if method == '2opt':
        return imp.two_opt(tour, **kwargs)
    elif method == "3opt":
        return imp.three_opt(tour, **kwargs)
    elif method == "kopt":
        return imp.k_opt(tour, **kwargs)
    elif method == "VOpt":
        return imp.v_opt(tour, **kwargs)
    elif method == "markov":
        return imp.markov(tour, **kwargs)

    return None

def show_tsp(dots, ax=None):
    assert dots.shape[1] == 2, "Only 2d display supported"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    dots = np.vstack([dots, dots[0, :]])
    ax.plot(dots[:, 1], -dots[:, 0], linewidth=1)

if __name__ == "__main__":
    np.random.seed(10)

    n = 100
    pts = (np.random.rand(n, 2) - .5) * 1000

    ordered_pts_nn = optimize(pts, method='nn')
    ordered_pts_nn_altered = improve(ordered_pts_nn)
    ordered_pts_nn_altered_2 = improve(ordered_pts_nn_altered)
    
    ordered_pts_rnn = optimize(pts, method='rnn')
    ordered_pts_rnn_altered = improve(ordered_pts_rnn)

    #ordered_pts_sa = optimize(pts, method='sa')
    #ordered_pts_sa_altered = improve(ordered_pts_sa)
    
    ordered_pts_ch = optimize(pts, method='christofides')
    ordered_pts_ch_altered = improve(ordered_pts_ch)

    print('plotting')
    fig = plt.figure()

    ax = fig.add_subplot(2,2,1)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_nn, ax)
    show_tsp(ordered_pts_nn_altered, ax)
    show_tsp(ordered_pts_nn_altered_2, ax)

    ax = fig.add_subplot(2,2,2)
    ax.scatter(pts[:, 1], -pts[:, 0])
    #show_tsp(ordered_pts_sa, ax)
    #show_tsp(ordered_pts_nn_altered, ax)

    ax = fig.add_subplot(2,2,3)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_ch, ax)
    show_tsp(ordered_pts_ch_altered, ax)

    ax = fig.add_subplot(2,2,4)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_rnn, ax)
    show_tsp(ordered_pts_rnn_altered, ax)

    plt.show()
