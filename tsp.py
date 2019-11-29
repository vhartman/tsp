import numpy as np
import random

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

def simulated_annealing(dots):
    def path_length(path):
        c = 0
        for i in range(len(path) - 1):
            c += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))

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
    while True:
        change = False
        for _ in range(len(path)):
            c = path_length(path) 
            print(c)

            start, end = get_rand_ind(len(path))
            r = np.random.rand()
            if r > 0.5:
                # transport
                path_cp = path[:start]
                path_cp.extend(path[end:])

                #print('transporting')
                #print(len(path))
                #print(len(path_cp))

                splice_pos = np.random.randint(0, len(path_cp))
                seg = path[start:end]
                #print(len(seg))
                #print(len(seg) + len(path_cp))
                #print()

                for i in range(len(seg)):
                    path_cp.insert(splice_pos + i, seg[i])
                
            else:
                #reverse
                path_cp = path[:]
                path_cp[start:end] = path_cp[end:start:-1]

                #print('reverse')
                #print(len(path_cp))
                #print(len(path))

            #print('lens')
            #print(len(path))
            #print(len(path_cp))
            #print()

            nc = path_length(path_cp)
            if nc < c:
                path = path_cp
                change = True
            else:
                diff = nc - c

                if np.exp(diff/T) > np.random.rand():
                    path = path_cp
                    change = True

        T = T-1
        print(T)
        if not change:
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
