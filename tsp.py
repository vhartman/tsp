import numpy as np
import kdtree

def nn(dots):
    ordered_dots = []
    not_visited = dots.tolist()

    current = not_visited[0]
    del not_visited[0]

    tree = kdtree.create(not_visited[1:])

    while tree.data is not None:
        nearest_info = tree.search_nn(current)
        nearest = nearest_info[0].data

        ordered_dots.append(nearest)
        current = nearest

        tree = tree.remove(current)
        
    return np.vstack(ordered_dots)

def tsp(dots, style='nn'):
    # https://developers.google.com/optimization/routing/tsp
    # mlrose

    if style == 'nn':
        return nn(dots)

    return None
