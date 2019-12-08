import numpy as np
import matplotlib.pyplot as plt

import random

import christofides
import kdtree

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
    if runs is None:
        runs = dots.shape[0]

    best_cost = None
    best_tour = None
    for i in range(runs):
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
        return repeated_nn(dots)
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

def alter_tour(dots):
    N = dots.shape[0]

    best_tour = dots.tolist()
    
    for l in range(N):
        for start in range(N):
            i = start
            j = (start + l) % N
            # check if alteration of tour decreases the cost
            current_cost = np.linalg.norm(np.array(best_tour[i-1]) - np.array(best_tour[i])) + np.linalg.norm(np.array(best_tour[j-1]) - np.array(best_tour[j]))
            cost = np.linalg.norm(np.array(best_tour[i-1]) - np.array(best_tour[j-1])) + np.linalg.norm(np.array(best_tour[j]) - np.array(best_tour[i]))

            if cost < current_cost:
                best_tour[i:j] = reversed(best_tour[i:j])

    return np.vstack(best_tour)

if __name__ == "__main__":
    np.random.seed(10)

    n = 100
    pts = np.random.rand(n, 2) * 1000

    print('nn')
    ordered_pts_nn = tsp(pts, style='nn')
    ordered_pts_nn_altered = alter_tour(ordered_pts_nn)
    ordered_pts_nn_altered_2 = alter_tour(ordered_pts_nn_altered)
    
    print('rnn')
    ordered_pts_rnn = tsp(pts, style='rnn')
    ordered_pts_rnn_altered = alter_tour(ordered_pts_rnn)

    print('sa')
    #ordered_pts_sa = tsp(pts, style='sa')
    #ordered_pts_sa_altered = alter_tour(ordered_pts_sa)
    
    print('chr')
    ordered_pts_ch = tsp(pts, style='christofides')
    ordered_pts_ch_altered = alter_tour(ordered_pts_ch)

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
