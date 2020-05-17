import numpy as np
import matplotlib.pyplot as plt

import random

import christofides
import util
import kdtree

import argparse

def path_length(path):
    c = 0
    for i in range(len(path)):
        c += np.linalg.norm(np.array(path[i-1]) - np.array(path[i]))

    return c

def nn(dots, start=0):
    assert start < dots.shape[0]

    not_visited = dots.tolist()

    current = not_visited[start]
    del not_visited[start]

    ordered_dots = [current]
    tree = kdtree.create(not_visited)

    while tree.data is not None:
        nearest_info = tree.search_nn(current)
        nearest = nearest_info[0].data

        ordered_dots.append(nearest)
        current = nearest

        tree = tree.remove(current)

    return np.vstack(ordered_dots)

def repeated_nn(dots, runs=None):
    N = dots.shape[0]

    if runs is None:
        runs = N

    runs = min([runs, N])

    start_pts = [i for i in range(N)]
    start_pts = np.random.choice(start_pts, runs, replace=False)

    best_cost = None
    best_tour = None
    for i in start_pts:
        tour = nn(dots, i)
        cost = path_length(tour)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour

def simulated_annealing(dots):
    def get_rand_ind(l):
        while True:
            ind = np.random.randint(0, l, 2)
            if ind[0] != ind[1]:
                break

        start = min(ind)
        end = max(ind)

        return start, end
    
    # generate random ordering
    path = dots.tolist()
    random.shuffle(path)

    best_yet = path
    best_cost_yet = None

    T = 1000
    alpha = 0.2
    while True:
        for _ in range(len(path)*10):
            c = path_length(path) 

            start, end = get_rand_ind(len(path))
            
            if True:
                path_cp = path[:]
                path_cp[start:end] = reversed(path[start:end])
            else:
                path_cp = path[:]
                path_cp[start] = path[end]
                path_cp[end] = path[start]

            nc = path_length(path_cp)
            if nc < c:
                path = path_cp
            else:
                diff = nc - c

                #print(T,np.exp(-diff/T), start, end)

                if np.exp(-diff/T) > np.random.rand():
                    path = path_cp

            if best_cost_yet is None or nc < best_cost_yet:
                best_cost_yet = nc
                best_yet = path

        T = T * (1 - alpha)
        if T <= 2.5:
            break

    return np.vstack(best_yet)

def nf(dots):
    # nearest fragment search
    return None

# match twice and stitch
def mts(dots):
    # initial matching: minimum weight matching
    # second matching: minimum weight matching with previous edges cost set to inf
    # stitching
    return None

def tsp(dots, style='nn'):
    # https://developers.google.com/optimization/routing/tsp
    # mlrose

    if style == 'nn':
        return nn(dots)
    elif style == 'rnn':
        return repeated_nn(dots, 10)
    elif style == 'sa':
        return simulated_annealing(dots)
    elif style == 'christofides':
        return christofides.christofides(dots)

    return None

def show_tsp(dots, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    dots = np.vstack([dots, dots[0, :]])
    ax.plot(dots[:, 1], -dots[:, 0], linewidth=1)

def alter_tour(dots, max_len=None, max_starts=None):
    N = dots.shape[0]

    if max_len is None:
        max_len = N

    starts = [i for i in range(N)]

    if max_starts is not None:
        if max_starts > N:
            max_starts = N

        starts = np.random.choice(starts, max_starts, replace=False)

    best_tour = dots.tolist()
    
    for l in range(max_len):
        for start in starts:
            i = start
            j = (start + l) % N
            # check if alteration of tour decreases the cost
            current_cost = np.linalg.norm(np.array(best_tour[i-1]) - np.array(best_tour[i])) \
                    + np.linalg.norm(np.array(best_tour[j-1]) - np.array(best_tour[j]))
            cost = np.linalg.norm(np.array(best_tour[i-1]) - np.array(best_tour[j-1])) \
                    + np.linalg.norm(np.array(best_tour[j]) - np.array(best_tour[i]))

            if cost < current_cost:
                best_tour[i:j] = reversed(best_tour[i:j])

    return np.vstack(best_tour)

def parse():
    parser = argparse.ArgumentParser(description='Stippling')

    parser.add_argument('--method', dest="method", default="all")
    parser.add_argument('--num', dest="num_dots", default=100, type=int)
   
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    np.random.seed(10)
    args = parse()

    n = args.num_dots
    pts = np.random.rand(n, 2) * 1000

    methods = args.method.split(',')
    x = len(methods)

    fig = plt.figure()
    for i, method in enumerate(methods):
        ordered_pts = tsp(pts, style=method)
        ordered_pts_altered = alter_tour(ordered_pts)
        ordered_pts_altered_2 = alter_tour(ordered_pts_altered)

        print(method)
        print('\t', path_length(ordered_pts))
        print('\t', path_length(ordered_pts_altered))
        print('\t', path_length(ordered_pts_altered_2))

        ax = fig.add_subplot(x, 3, i*3+1)
        ax.scatter(pts[:, 1], -pts[:, 0])
        ax.set_aspect('equal')
        show_tsp(ordered_pts, ax)

        ax = fig.add_subplot(x, 3, i*3+2)
        ax.scatter(pts[:, 1], -pts[:, 0])
        ax.set_aspect('equal')
        show_tsp(ordered_pts_altered, ax)
        
        ax = fig.add_subplot(x, 3, i*3+3)
        ax.scatter(pts[:, 1], -pts[:, 0])
        ax.set_aspect('equal')
        show_tsp(ordered_pts_altered_2, ax)
    
    plt.show()
