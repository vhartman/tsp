import numpy as np

def get_rand_ind(l):
    while True:
        ind = np.random.randint(0, l, 2)
        if ind[0] != ind[1]:
            break

    start = min(ind)
    end = max(ind)

    return start, end

def path_length(path, cost=None):
    if cost is None:
        cost = np.linalg.norm

    c = 0
    for i in range(len(path)):
        c += cost(np.array(path[i-1]) - np.array(path[i]))

    return c


