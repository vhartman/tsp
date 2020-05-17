import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment

class N:
    def __init__(self, i):
        self.links = []
        self.ind = i

def christofides(dots):
    d = np.zeros((len(dots), len(dots)))
    for i in range(len(dots)):
        for j in range(i+1, len(dots)):
            d[i, j] = np.linalg.norm(dots[i] - dots[j])

    X = csr_matrix(d)
    Tcsr = minimum_spanning_tree(X)

    # get points with odd degree 
    np.set_printoptions(precision=1)
    tcsr_arr = Tcsr.toarray()
    O = np.argwhere((((tcsr_arr > 0) + (tcsr_arr.T > 0)).sum(axis=1)) % 2).flatten()

    #f = plt.figure()
    #ax = f.add_subplot(1, 2, 1)
    G = []
    for i in range(dots.shape[0]):
        G.append(N(i))

    for i in range(tcsr_arr.shape[0]):
        for j in range(i+1, tcsr_arr.shape[0]):
            if tcsr_arr[i, j] > 0:
                G[i].links.append(j)
                G[j].links.append(i)
                #ax.plot([dots[i, 0], dots[j, 0]], [dots[i, 1], dots[j, 1]], c='black', alpha=0.5)

    #ax.scatter(*dots.T)

    O_d = d[np.ix_(O, O)]
    O_d += O_d.T
    O_d += np.eye(len(O_d)) * 100000000000

    p = []
    matched = []
    for a, i in enumerate(O):
        m = 10000000000
        ind = -1
        if i in matched:
            continue

        for j in O[a+1:]:
            if d[i, j] < m and j not in matched:
                ind = j
                m = d[i, j]

        matched.append(ind)
        p.append((i, ind))
        
    for i, j in p:
        G[i].links.append(j)
        G[j].links.append(i)
    
    #for i in G:
    #    print(i.ind, len(i.links))
    #print()

    def dfs(graph, start):
        op = [start]
        visited = []

        cnt = 0
        while True:
            current = op[0]
            visited.append(current)
            
            for n in graph[current].links:
                if n not in visited and n not in op:
                    op.append(n)

            del op[0]

            cnt += 1
            if len(op) == 0:
                break

        return cnt

    # form eulerian circuit using the previous results using fleurys algorithm
    # pick a random start in the graph
    node = G[0]
    path = [0]
    while True:
        # select node
        if len(node.links) == 1:
            j = node.links[0]
        else:
            # check which child nodes are bridges
            for i, l in enumerate(node.links):
                #f = plt.figure()
                #ax = f.add_subplot(1,2,1)
                #for h in G:
                #    for g in h.links:
                #        ax.plot([dots[h.ind, 0], dots[g, 0]], [dots[h.ind, 1], dots[g, 1]], c='black', alpha=0.1)

                cnt = dfs(G, node.ind)
                
                G[l].links.remove(node.ind)
                G[node.ind].links.remove(l)

                #ax = f.add_subplot(1,2,2)
                #for h in G:
                #    for g in h.links:
                #        ax.plot([dots[h.ind, 0], dots[g, 0]], [dots[h.ind, 1], dots[g, 1]], c='black', alpha=0.1)
                #ax.scatter(dots[node.ind, 0], dots[node.ind, 1])
                #ax.scatter(dots[l, 0], dots[l, 1])
                cnt_rm = dfs(G, node.ind)

                G[l].links.append(node.ind)
                G[node.ind].links.append(l)

                #plt.show()

                #print(node.ind)
                #print(cnt)
                #print(cnt_rm)
                #print(l, node.links)
                #print(l, G[l].links)
                #print()
                if cnt == cnt_rm:
                    j = l
                    break

        path.append(j)

        # move to next node
        # remove edge that was just traversed

        ind = node.ind
        node.links.remove(j)
        #print(node.ind)
        #print(node.links)

        node = G[j]
        node.links.remove(ind)

        #print(node.ind)
        #print(node.links)
        #print()

        if len(node.links) == 0:
            break
    
    #ax = f.add_subplot(1, 2, 2)
    #for o in range(len(path)):
    #    i = path[o-1]
    #    j = path[o]
    #    ax.plot([dots[i, 0], dots[j, 0]], [dots[i, 1], dots[j, 1]],c='black', alpha=0.5)

    #plt.show()

    # skip repeated vertices
    visited = []
    ordered_dots = []
    for p in path:
        if p not in visited:
            visited.append(p)
            ordered_dots.append(dots[p, :])

    return np.vstack(ordered_dots)
