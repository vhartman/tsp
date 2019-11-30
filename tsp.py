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

        T = T * (1 - alpha)
        if T <= 2.5:
            break

    return np.vstack(path)


def tsp(dots, style='nn'):
    # https://developers.google.com/optimization/routing/tsp
    # mlrose

    if style == 'nn':
        return nn(dots)
    elif style == 'sa':
        return simulated_annealing(dots)

    return None

def show_tsp(dots, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    dots = np.vstack([dots, dots[0, :]])
    ax.plot(dots[:, 1], -dots[:, 0], linewidth=1)

if __name__ == "__main__":
    np.random.seed(10)

    n = 100
    pts = np.random.rand(n, 2) * 1000

    ordered_pts_nn = tsp(pts, style='nn')
    ordered_pts_sa = tsp(pts, style='sa')

    fig = plt.figure()

    ax = fig.add_subplot(1,2,1)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_nn, ax)

    ax = fig.add_subplot(1,2,2)
    ax.scatter(pts[:, 1], -pts[:, 0])
    show_tsp(ordered_pts_sa, ax)

    plt.show()
