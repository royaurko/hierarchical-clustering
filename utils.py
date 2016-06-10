from __future__ import print_function
from __future__ import division
import random
import math
import csv
import numpy as np
import networkx as nx
import ast
import matplotlib.pyplot as plt
from munkres import Munkres


def sample(data, target, num_samples):
    n = data.shape[0]
    indices = set()
    while len(indices) < num_samples:
        i = random.randint(0, n-1)
        indices.add(i)
    i = indices.pop()
    sampled_data = data[i]
    sampled_target = target[i]
    while indices:
        i = indices.pop()
        sampled_data = np.vstack((sampled_data, data[i]))
        sampled_target = np.vstack((sampled_target, target[i]))
    sampled_target = np.reshape(sampled_target, (sampled_target.shape[0], ))
    return sampled_data, sampled_target


def complete_ultrametric(d):
    universe_pairs = d.keys()
    universe = list(set([i for pair in universe_pairs for i in pair]))
    n = len(universe)
    for i in range(n):
        if (i, i) not in d:
            d[i, i] = 0
        for j in range(i):
            if (i, j) not in d:
                d[i, j] = d[j, i]


def error(cluster, target_cluster):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    k = len(set(target_cluster))
    n = len(target_cluster)
    C = []
    T = []
    for i in range(1, k+1):
        tmp = {j for j in range(n) if cluster[j] == i}
        C.append(tmp)
        tmp = {j for j in range(n) if target_cluster[j] == i}
        T.append(tmp)
    M = []
    for i in range(k):
        M.append([0]*k)
    testM = []
    for i in range(k):
        testM.append([0]*k)
    for i in range(k):
        for j in range(k):
            M[i][j] = len(C[i].difference(T[j]))
            testM[i][j] = len(T[j].difference(C[i]))
    m = Munkres()
    indexes = m.compute(M)
    total = 0
    for row, col in indexes:
        value = M[row][col]
        total += value
    indexes2 = m.compute(testM)
    total2 = 0
    for row, col in indexes2:
        value = testM[row][col]
        total2 += value
    err =  float(total)/float(n)
    return err


def prune(L, target, k, labels):
    if k > len(L):
        # Not enough clusters
        return
    if len(L) == 0 and k == 0:
        return error(labels, target), labels
    if k == 0 and len(L) > 0:
        return
    item = L[0]
    disjoint_elems = []
    for s in L:
        if s & item:
            continue
        else:
            disjoint_elems.append(s)
    non_incl_err = prune(L[1:], target, k, labels)
    for i in item:
        labels[i] = k
    incl_err = prune(disjoint_elems, target, k-1, labels)
    if incl_err is None:
        return non_incl_err
    if non_incl_err is None:
        return incl_err
    if incl_err[0] < non_incl_err[0]:
        return incl_err
    else:
        return non_incl_err


def get_total(m):
    # return total similarity function
    n = m._n
    s = 0
    for i in xrange(n):
        for j in xrange(i + 1, n):
            for t in xrange(1, n):
                s += m._obj[i, j, t]
    return s


def draw(G, target, n, tree_name):
    # Draw the hierarchy
    pos = hierarchy_pos(G, str(range(n)))
    nodes = nx.nodes(G)
    edges = nx.edges(G)
    labels = {}
    colors = ['r', 'g', 'b', 'm']
    for node in nodes:
        node_list = ast.literal_eval(node)
        if len(node_list) == 1:
            labels[node] = ''
            #labels[node] = str(node_list[0] + 1)
            labels[node] = "$x_{" + str(node_list[0]) + "}$"
            color = colors[int(target[node_list[0]])]
        else:
            color = 'w'
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[node], node_color=color)
    nx.draw_networkx_edges(G, pos=pos)
    # nx.draw_networkx_labels(G, pos=pos, labels=labels)
    for node in labels:
        x, y = pos[node]
        plt.text(x, y - 0.05, s=labels[node], fontsize=18, horizontalalignment='center')
    plt.axis('off')
    plt.savefig(tree_name)


def check_ultrametric(d):
    # Check if ultrametric
    universe_pairs = d.keys()
    universe = list(set([i for pair in universe_pairs for i in pair]))
    n = len(universe)
    flag = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if int(round(d[i, k])) > max(int(round(d[i, j])), int(round(d[j, k]))):
                    print('i = {0}, j = {1}, k = {2}'.format(i, j, k))
                    print('d[i, k] = {0}, d[i, j] = {1}, d[j, k] = {2}'.format(d[i, k], d[i, j], d[j, k]))
                    flag = False
                if int(round(d[i, j])) > max(int(round(d[i, k])), int(round(d[j, k]))):
                    print('i = {0}, j = {1}, k = {2}'.format(i, j, k))
                    print('d[i, k] = {0}, d[i, j] = {1}, d[j, k] = {2}'.format(d[i, k], d[i, j], d[j, k]))
                    flag = False
                if int(round(d[j, k])) > max(int(round(d[i, j])), int(round(d[i, k]))):
                    print('i = {0}, j = {1}, k = {2}'.format(i, j, k))
                    print('d[i, k] = {0}, d[i, j] = {1}, d[j, k] = {2}'.format(d[i, k], d[i, j], d[j, k]))
                    flag = False
    return flag


def inverse_ultrametric(d, inverse_f):
    # If d is the f-image of an ultrametric, invert it
    universe_pairs = d.keys()
    universe = list(set([i for pair in universe_pairs for i in pair]))
    universe.sort()
    n = len(universe)
    for i in range(n):
        ind1 = universe[i]
        for j in range(i + 1, n):
            ind2 = universe[j]
            inverse_d = inverse_f(d[ind1, ind2])
            d[ind1, ind2] = int(inverse_f(d[ind1, ind2]))


def partition(d):
    # Partition elements of d into sub-ultrametrics where distances are < n - 1
    universe_pairs = d.keys()
    universe = list(set([j for pair in universe_pairs for j in pair]))
    universe.sort()
    equiv_classes = []
    n = len(universe)
    visited = [0]*n
    for j in range(n):
        if not visited[j]:
            visited[j] = 1
            c = [universe[j]]
            for k in range(j + 1, n):
                if d[universe[j], universe[k]] < n - 1:
                    c.append(universe[k])
                    visited[k] = 1
            equiv_classes.append(c)
    return equiv_classes


def build_laminar_list(d):
    # Build laminar list from the ultrametric
    universe_pairs = d.keys()
    universe = list(set([i for pair in universe_pairs for i in pair]))
    n = len(universe)
    if n == 1:
        return []
    laminar_list = []
    # Take elements at distance <= n - 1, forms equivalence class
    equiv_classes = partition(d)
    for c in equiv_classes:
        laminar_list.append(set(c))
        restricted_d = {}
        for i in range(len(c)):
            restricted_d[c[i], c[i]] = 0
            for j in range(i + 1, len(c)):
                restricted_d[c[i], c[j]] = d[c[i], c[j]]
                restricted_d[c[j], c[i]] = d[c[j], c[i]]
        l = build_laminar_list(restricted_d)
        laminar_list += l
    return laminar_list


def test_laminar(L):
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if set(L[i]) & set(L[j]):
                if not set(L[i]) <= set(L[j]) and not set(L[j]) <= set(L[i]):
                    print('a = ', L[i])
                    print('b = ', L[j])
                    return False
    return True


def prepare_data(f, l):
    # Return data and target as numpy arrays
    print('Preparing data')
    reader = csv.reader(open(f), skipinitialspace=True, delimiter=',')
    data = []
    target = []
    for row in reader:
        if row:
            if l != -1:
                label = row[l]
                row.pop(l)
                data.append(row)
                target.append(label)
            else:
                data.append(row)
    data = np.array(data, dtype=float)
    if l != -1:
        labels = set(target)
        label_to_idx = {v: i for i, v in enumerate(labels)}
        target = np.array([label_to_idx[i] for i in target], dtype=int)
    else:
        target = np.zeros(shape=data.shape[0])
    return data, target


def hierarchy_pos(G, root, width=1.0, vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
                  pos = None, parent = None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.'''
    if pos == None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = G.neighbors(root)
    if parent != None:
        neighbors.remove(parent)
    if len(neighbors)!=0:
        dx = width/len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width = dx, vert_gap = vert_gap,
                                vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                parent = root)
    return pos


def build_hierarchy(d):
    # Build the hierarchy corresponding to ultrametric d
    G = nx.Graph()
    universe_pairs = d.keys()
    universe = list(set([i for pair in universe_pairs for i in pair]))
    n = len(universe)
    universe.sort()
    G.add_node(str(universe))
    if n == 1:
        return G
    classes = partition(d)
    for c in classes:
        restricted_d = {}
        for i in range(len(c)):
            restricted_d[c[i], c[i]] = 0
            for j in range(i + 1, len(c)):
                restricted_d[c[i], c[j]] = d[c[i], c[j]]
                restricted_d[c[j], c[i]] = d[c[j], c[i]]
        H = build_hierarchy(restricted_d)
        G.add_nodes_from(H)
        G.add_edges_from(H.edges())
        G.add_edge(str(universe), str(c))
    return G


def linear(x):
    return x


def inverse_linear(x):
    return x


def quadratic(x):
    return pow(x, 2)


def inverse_quadratic(x):
    return pow(x, 0.5)


def cubic(x):
    return pow(x, 3)


def inverse_cubic(x):
    return pow(x, (1.0/3))


def exponential(x):
    return math.exp(x) - 1


def inverse_exponential(x):
    return logarithm(x)


def logarithm(x):
    return math.log(1 + x)


def inverse_logarithm(x):
    return exponential(x)


def get_cost(m, d):
    n = m._n
    cost = 0
    for i in xrange(n):
        for j in xrange(i + 1, n):
            for t in xrange(1, n):
                if d[i, j] >= t:
                    cost += m._obj[i, j, t]
    return cost


