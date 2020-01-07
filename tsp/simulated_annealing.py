import numpy as np
import random

import tsp.util as util

def simulated_annealing(dots):
    # generate random ordering
    path = dots.tolist()
    random.shuffle(path)

    best_yet = path
    best_cost_yet = None

    T = 1000
    alpha = 0.2
    while True:
        for _ in range(len(path)*10):
            c = util.path_length(path) 

            start, end = util.get_rand_ind(len(path))
            
            if True:
                path_cp = path[:]
                path_cp[start:end] = reversed(path[start:end])
            else:
                path_cp = path[:]
                path_cp[start] = path[end]
                path_cp[end] = path[start]

            nc = util.path_length(path_cp)
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

