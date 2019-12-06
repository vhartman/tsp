import numpy as np
import matplotlib.pyplot as plt

import random

import kdtree

def nn(dots):
    not_visited = dots.tolist()

    current = not_visited[0]
    del not_visited[0]

    ordered_dots = [current]
    tree = kdtree.create(not_visited)

    while tree.data is not None:
        nearest_info = tree.search_nn(current)
        nearest = nearest_info[0].data

        ordered_dots.append(nearest)
        current = nearest

        tree = tree.remove(current)

    return np.vstack(ordered_dots)

def simulated_annealing(dots):
    def path_length(path):
        c = 0
        for i in range(len(path)):
            c += np.linalg.norm(np.array(path[i-1]) - np.array(path[i]))

        return c

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
                seg = path[start:end]
                seg = seg[::-1]
                path_cp[start:end] = seg
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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment

def christofides(dots):
    d = np.zeros((len(dots), len(dots)))
    for i in range(len(dots)):
        for j in range(i+1, len(dots)):
            d[i, j] = np.linalg.norm(dots[i] - dots[j])

    X = csr_matrix(d)
    Tcsr = minimum_spanning_tree(X)

    # get points with odd degree 
    tcsr_arr = Tcsr.toarray()
    O = np.argwhere((tcsr_arr > 0).sum(axis=1) % 2).flatten()

    O_d = d[np.ix_(O, O)]
    O_d += O_d.T
    O_d += np.eye(len(O_d)) * 100000000

    r, c = linear_sum_assignment(O_d)

    # form eulerian circuit using the previous results using fleurys algorithm
    # pick a random start in the graph
    node = None
    while True:
        # check which child nodes are bridges
        # pick one of the non-bridges
        # move to next node
        # remove edge that was just traversed
        # add to path
        break
    
    # skip repeated vertices
    node = None
    for i in range(num_nodes):
        # pick one of the paths leading away from the node (non-traversed paths)
        # if resulting node is visited, pick from outgoing edges and repeat
        # mark as visited
        # mark outgoing as traversed

    return None

def tsp(dots, style='nn'):
    # https://developers.google.com/optimization/routing/tsp
    # mlrose

    if style == 'nn':
        return nn(dots)
    elif style == 'sa':
        return simulated_annealing(dots)
    elif style == 'christofides':
        return christofides(dots)

    return None

def show_tsp(dots, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    dots = np.vstack([dots, dots[0, :]])
    ax.plot(dots[:, 1], -dots[:, 0], linewidth=1)

if __name__ == "__main__":
    np.random.seed(10)

    n = 10
    pts = np.random.rand(n, 2) * 1000

    #ordered_pts_nn = tsp(pts, style='nn')
    #ordered_pts_sa = tsp(pts, style='sa')
    ordered_pts_ch = tsp(pts, style='christofides')

    exit()

    fig = plt.figure()

    ax = fig.add_subplot(2,2,1)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_nn, ax)

    ax = fig.add_subplot(2,2,2)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_sa, ax)

    ax = fig.add_subplot(2,2,3)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_ch, ax)

    plt.show()
