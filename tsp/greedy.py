import numpy as np

import tsp.kdtree as kdtree
import tsp.util as util

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
        cost = util.path_length(tour)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour

def nf(dots):
    # nearest fragment search
    segments = []

    while True:
        # add segments
        break
    return None
