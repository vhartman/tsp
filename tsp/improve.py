import numpy as np

def two_opt(tour, max_len=None, max_starts=None):
    N = tour.shape[0]

    if max_len is None:
        max_len = N

    starts = [i for i in range(N)]

    if max_starts is not None:
        if max_starts > N:
            max_starts = N

        starts = np.random.choice(starts, max_starts, replace=False)

    best_tour = tour.tolist()
    
    for l in range(max_len):
        for start in starts:
            i = start
            j = (start + l) % N

            # check if alteration of tour decreases the cost
            # TODO: make general for k-opt
            current_cost = np.linalg.norm(np.array(best_tour[i-1]) - np.array(best_tour[i]))\
                           + np.linalg.norm(np.array(best_tour[j-1]) - np.array(best_tour[j]))
            cost = np.linalg.norm(np.array(best_tour[i-1]) - np.array(best_tour[j-1]))\
                   + np.linalg.norm(np.array(best_tour[j]) - np.array(best_tour[i]))

            if cost < current_cost:
                best_tour[i:j] = reversed(best_tour[i:j])

    return np.vstack(best_tour)

def k_opt(tour, k):
    # TODO
    def segment_cost(points):
        pass

    pass
