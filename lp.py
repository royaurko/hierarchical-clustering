from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
from scipy.spatial.distance import pdist, cosine
import scipy.cluster.hierarchy as hac
import time
import itertools
import math
from gurobipy import *
import utils
from sklearn.cluster import KMeans
import random


def fast_spreading_constraints(m, num_t=5):
    # Fast spreading constraint using S = V and sampling t
    n = m._n
    flag = False
    # Randomly sample num_t many t's to add constraints
    for i in xrange(n):
        for _ in xrange(num_t):
            t = random.randint(1, n-1)
            # for t in xrange(1, n):
            pre_var_list = [m._vars[j, i, t] for j in xrange(i)]
            post_var_list = [m._vars[i, j, t] for j in xrange(i + 1, n)]
            var_list = pre_var_list + post_var_list
            s = sum([y.X for y in var_list])
            if s < n - t:
                m.addConstr(sum(var_list) >= n - t)
                flag = True
    return flag


def spreading_constraints(m):
    # Constraints to make ultrametric non-trivial
    n = m._n
    # Get variables
    tol = 1e-5
    flag = False
    for i in xrange(n):
        for t in xrange(1, n):
            pre_var_list = [(m._vars[j, i, t], j) for j in xrange(i)]
            post_var_list = [(m._vars[i, j, t], j) for j in xrange(i + 1, n)]
            var_list = pre_var_list + post_var_list
            solution = map(lambda y: (y[0].X, y[1]), var_list)
            solution.sort(key=lambda x: x[0])
            s = 0
            expr = 0
            # k only needs to be n
            for k in xrange(len(solution)):
                # k iterates over the sizes of the set S
                s += solution[k][0]
                j = solution[k][1]
                if j > i:
                    expr += m._vars[i, j, t]
                else:
                    expr += m._vars[j, i, t]
                if abs(s - (k + 2 - t)) > tol and s < k + 2 - t:
                    m.addConstr(expr >= k + 2 - t)
                    flag = True
    return flag


def triangle_constraints(m):
    # Lazy triangle inequality
    n = m._n
    tol = 1e-5
    flag = False
    for i in xrange(n):
        for j in xrange(i + 1, n):
            for t in xrange(1, n):
                v_ijt = m._vars[i, j, t]
                x_ijt = v_ijt.X
                for k in xrange(j + 1, n):
                    v_ikt = m._vars[i, k, t]
                    v_jkt = m._vars[j, k, t]
                    x_ikt = v_ikt.X
                    x_jkt = v_jkt.X
                    if x_ijt > x_ikt + x_jkt and abs(x_ijt - x_ikt - x_jkt) > tol:
                        flag = True
                        m.addConstr(v_ikt + v_jkt >= v_ijt)
                    if x_ikt > x_ijt + x_jkt and abs(x_ikt - x_ijt - x_jkt) > tol:
                        flag = True
                        m.addConstr(v_ijt + v_jkt >= v_ikt)
                    if x_jkt > x_ijt + x_ikt and abs(x_jkt - x_ijt - x_ikt) > tol:
                        flag = True
                        m.addConstr(v_ijt + v_ikt >= v_jkt)
    return flag


def add_triangle_spreading_constraints(m):
    # Add trianlge inequality and spreading constraints explicitly
    n = m._n
    start = time.time()
    for t in xrange(1, n):
        for i in xrange(n):
            # Add spreading constraint
            pre_var_list = [m._vars[j, i, t] for j in xrange(i)]
            post_var_list = [m._vars[i, j, t] for j in xrange(i + 1, n)]
            var_list = pre_var_list + post_var_list
            cname = "spreading_{0}_{1}".format(i, t)
            m.addConstr(sum(var_list) >= n - t, cname)
            # Add triangle inequality
            for j in xrange(i + 1, n):
                for k in xrange(j + 1, n):
                    cname1 = 'triangle_{0}_{1}_{2}_{3}'.format(i, j, k, t)
                    cname2 = 'triangle_{0}_{1}_{2}_{3}'.format(i, k, j, t)
                    cname3 = 'triangle_{0}_{1}_{2}_{3}'.format(j, k, i, t)
                    m.addConstr(m._vars[i, j, t] <= m._vars[i, k, t] + m._vars[j, k, t], cname1)
                    m.addConstr(m._vars[i, k, t] <= m._vars[i, j, t] + m._vars[j, k, t], cname2)
                    m.addConstr(m._vars[j, k, t] <= m._vars[i, j, t] + m._vars[i, k, t], cname3)
    m.update()
    end = time.time()
    print('Time to add triangle + spreading constraints = {0}'.format(end - start))


def add_triangle_constraints(m):
    # Add trianlge inequality explicitly
    n = m._n
    start = time.time()
    for t in xrange(1, n):
        for i in xrange(n):
            for j in xrange(i + 1, n):
                for k in xrange(j + 1, n):
                    cname1 = 'triangle_{0}_{1}_{2}_{3}'.format(i, j, k, t)
                    cname2 = 'triangle_{0}_{1}_{2}_{3}'.format(i, k, j, t)
                    cname3 = 'triangle_{0}_{1}_{2}_{3}'.format(j, k, i, t)
                    m.addConstr(m._vars[i, j, t] <= m._vars[i, k, t] + m._vars[j, k, t], cname1)
                    m.addConstr(m._vars[i, k, t] <= m._vars[i, j, t] + m._vars[j, k, t], cname2)
                    m.addConstr(m._vars[j, k, t] <= m._vars[i, j, t] + m._vars[i, k, t], cname3)
    m.update()
    end = time.time()
    print('Time to add triangle constraints = {0}'.format(end - start))


def separation_oracle(m, triangle):
    # First check if triangle constraints are violated
    flag_triangle = False
    if triangle:
        flag_triangle = triangle_constraints(m)
    flag_separation = fast_spreading_constraints(m)
    # flag_separation = spreading_constraints(m)
    return flag_triangle or flag_separation


def add_variables(data, m, kernel, f):
    # Add variables to m
    print('Adding variables to model')
    v = {}
    w = {}
    start = time.time()
    counter = 0
    n = m._n
    for i in xrange(n):
        for j in xrange(i + 1, n):
            for t in xrange(1, n):
                # w[i, j, t] is the kernel function
                if kernel == 'gaussian':
                    s = 100000
                    dist = np.linalg.norm(data[i] - data[j])
                    w[i, j, t] = (f(t) - f(t-1)) * np.exp(- (dist ** 2)/(s **2))
                else:
                    w[i, j, t] = (f(t) - f(t-1)) * (1 - cosine(data[i], data[j]))
                vname = 'x_{0}_{1}_{2}'.format(i, j, t)
                v[i, j, t] = m.addVar(lb=0.0, ub=1.0, obj=w[i, j, t], vtype=GRB.CONTINUOUS, name=vname)
                counter += 1
    m.update()
    m._vars = v
    m._obj = w
    end = time.time()
    print('Time to add variables = {0:.2f}s'.format(end - start))


def add_layer_constraints(m):
    # add layer constraints
    n = m._n
    start = time.time()
    for t in xrange(1, n-1):
        for i in xrange(n):
            for j in xrange(i + 1, n):
                cname = 'layer_{0}_{1}_{2}'.format(i, j, t)
                m.addConstr(m._vars[i, j, t] - m._vars[i, j, t + 1] >= 0, cname)
    m.update()
    end = time.time()
    print('Time to add layer constraints = {0}'.format(end - start))


def check_triangle_constraints(m):
    # Check if solution satisfies triangle inequality
    n = m._n
    tol = 1e-5
    for i in xrange(n):
        for j in xrange(i + 1, n):
            for t in xrange(1, n):
                v_ijt = m._vars[i, j, t]
                x_ijt = v_ijt.X
                for k in xrange(j + 1, n):
                    v_ikt = m._vars[i, k, t]
                    v_jkt = m._vars[j, k, t]
                    x_ikt = v_ikt.X
                    x_jkt = v_jkt.X
                    if x_ijt > x_ikt + x_jkt and abs(x_ijt - x_ikt - x_jkt) > tol:
                        return False
                    if x_ikt > x_ijt + x_jkt and abs(x_ikt - x_ijt - x_jkt) > tol:
                        return False
                    if x_jkt > x_ijt + x_ikt and abs(x_jkt - x_ijt - x_ikt) > tol:
                        return False
    return True


def check_spreading_constraints(m):
    # Iterate over subsets
    n = m._n
    tol = 1e-5
    for i in xrange(n):
        for t in xrange(1, n):
            for l in xrange(1, n):
                # l + 1 is the length of the subset
                elems = [j for j in xrange(n) if j != i]
                subsets = itertools.combinations(elems, l)
                for s in subsets:
                    # s is a tuple
                    v = sum(map(lambda j: m._vars[i, j, t].X if i < j else m._vars[j, i, t].X, s))
                    if v < l + 1 - t and abs(v - l - 1 + t) > tol:
                        print('i = {0}, t = {1}, s = {2}, v = {3}'.format(i, t, s, v))
                        return False
    return True


def init_model(data, kernel, triangle, f):
    # Initiliaze Gurobi model
    m = Model()
    m._n = data.shape[0]
    # Add variables
    add_variables(data, m, kernel, f)
    # Add layer constraints
    add_layer_constraints(m)
    if not triangle:
        add_triangle_spreading_constraints(m)
    m.modelSense = GRB.MINIMIZE
    m.params.LazyConstraints = 1
    return m


def round_solution(m, solution_dict, eps):
    # Round the solutions to the LP
    n = m._n
    C_prev = [set(range(n))]
    x = {}
    tol = 1e-5
    for i in range(n):
        for j in range(i + 1, n):
            for t in range(1, n):
                x[i, j, t] = 1
    delta = float(eps)/float(1 + eps)
    for t in xrange(n-1, 0, -1):
        # run while C not empty
        C = []
        visited = [0]*n
        gamma_t = gamma(m, solution_dict, t)
        while C_prev:
            U = C_prev.pop()
            if len(U) < (1 + eps)*t:
                ulist = list(U)
                for j in range(len(ulist)):
                    ind1 = ulist[j]
                    visited[ind1] = 1
                    for k in range(j + 1, len(ulist)):
                        ind2 = ulist[k]
                        if ind1 < ind2:
                            x[ind1, ind2, t] = 0
                        else:
                            x[ind2, ind1, t] = 0
                C.append(U)
            else:
                while U:
                    i = U.pop()
                    v_delta = gamma_t + volume(m, solution_dict, i, delta, t)
                    v_zero = gamma_t

                    # Helpful prints
                    # print('i = {0}, U = {1}, t = {2}'.format(i, U, t))
                    # print('v_delta = {0}, v_zero = {1}'.format(v_delta, v_zero))

                    try:
                        threshold = math.log(v_delta/v_zero)
                        threshold /= delta
                    except:
                        print("v_delta = {}, v_zero = {}".format(v_delta, v_zero))
                    pre_var_list = [(solution_dict[j, i, t], j) for j in range(i)]
                    post_var_list = [(solution_dict[i, j, t], j) for j in range(i + 1, n)]
                    var_list = pre_var_list + post_var_list
                    var_list.sort(key=lambda x: x[0])
                    least_r = 0
                    least_ind = 0
                    for j in xrange(len(var_list)):
                        if abs(var_list[j][0] - 0) > tol:
                            least_r = var_list[j][0]/2
                            least_ind = j
                            break
                    exp = expansion(m, solution_dict, i, least_r, t)
                    while exp > threshold and least_ind < len(var_list)-1:
                        print('expansion = ', exp)
                        least_r = (var_list[least_ind][0] + var_list[least_ind + 1][0])/2
                        least_ind += 1
                        exp = expansion(m, solution_dict, i, least_r, t)
                    ball = get_ball(m, solution_dict, i, least_r, t)

                    # Helpful prints
                    # print('i = {0}, t = {1}, ball = {2}'.format(i, t, ball))

                    # Assert claim of theorem 1
                    assert(len(ball) <= (1 + eps) * t)
                    disjoint_ball = set()

                    for index in ball:
                        if not visited[index]:
                            disjoint_ball.add(index)
                            visited[index] = 1

                    ball_list = list(disjoint_ball)
                    # Set x[i, j, t] = 0 for elements in disjoint_ball
                    for j in xrange(len(ball_list)):
                        ind1 = ball_list[j]
                        for k in xrange(j + 1, len(ball_list)):
                            ind2 = ball_list[k]
                            if ind1 < ind2:
                                x[ind1, ind2, t] = 0
                            else:
                                x[ind2, ind1, t] = 0
                    U = U.difference(ball)
                    C.append(disjoint_ball)
        C_prev = C
        # Helpful prints
        print('t = {0}, C = {1}'.format(t, C))
    return x


def check_disjoint(C):
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            if C[i] & C[j]:
                print('set1 = {0}, set2 = {1}'.format(C[i], C[j]))
                return False
    return True


def get_ultrametric_from_lp(m, solution_dict, eps, f):
    # Get ultrametric
    n = m._n
    x = round_solution(m, solution_dict, eps)
    d = {}
    for i in xrange(n):
        for j in xrange(i + 1, n):
            d[i, j] = f(1) - f(0)
            for t in xrange(2, n):
                rounded_t = int(math.floor(t/(1 + eps)))
                d[i, j] += (f(t) - f(t-1)) * x[i, j, rounded_t]
    return d


def get_ball(m, solution_dict, i, r, t):
    # Get ball B(i, r, t)
    ball = set([])
    n = m._n
    for j in xrange(n):
        if j < i:
            dist = solution_dict[j, i, t]
        elif j == i:
            dist = 0
        else:
            dist = solution_dict[i, j, t]
        if dist < r:
            ball.add(j)
    return ball


def gamma(m, solution_dict, t):
    # Get gamma_t
    gamma_t = 0
    n = m._n
    for i in range(n):
        for j in range(i + 1, n):
            gamma_t += m._obj[i, j, t] * solution_dict[i, j, t]
    return gamma_t


def volume(m, solution_dict, i, r, t):
    # Get volume of ball B(i, r, t) without the affine shift
    n = m._n
    internal_vol = 0
    ball = get_ball(m, solution_dict, i, r, t)
    if not ball:
        return 0
    ball_list = list(ball)
    ball_list.sort()
    for ind1 in xrange(len(ball_list)):
        for ind2 in xrange(ind1 + 1, len(ball_list)):
            internal_vol += m._obj[ball_list[ind1], ball_list[ind2], t] * solution_dict[ball_list[ind1], ball_list[ind2], t]
    # Get crossing volumes
    complement = [j for j in xrange(n) if j not in ball]
    crossing_vol = 0
    for ind1 in xrange(len(ball_list)):
        for ind2 in xrange(len(complement)):
            j = ball_list[ind1]
            k = complement[ind2]
            if j < k:
                obj = m._obj[j, k, t]
            else:
                obj = m._obj[k, j, t]
            if i < j:
                diff = r - solution_dict[i, j, t]
            elif i == j:
                diff = r
            else:
                diff = r - solution_dict[j, i, t]
            crossing_vol += obj * diff
    return internal_vol + crossing_vol


def boundary(m, s, t):
    # Get boundary of set s
    n = m._n
    complement = [j for j in xrange(n) if j not in s]
    b = 0
    for i in set(s):
        for j in complement:
            if i < j:
                b += m._obj[i, j, t]
            else:
                b += m._obj[j, i, t]
    return b


def expansion(m, solution_dict, i, r, t):
    # Get expansion of set s
    ball = get_ball(m, solution_dict, i, r, t)
    b = boundary(m, ball, t)
    v = volume(m, solution_dict, i, r, t)
    return b/(gamma(m, solution_dict, t) + v)


def read_solution(m, solution_file):
    with open(solution_file) as f:
        lines = f.readlines()
    solution_dict = {}
    lines.pop(0)
    for line in lines:
        c = map(lambda x: x.rstrip('\n'), line.split(' '))
        indices = c[0].split('_')
        i = int(indices[1])
        j = int(indices[2])
        t = int(indices[3])
        solution_dict[i, j, t] = float(c[1])
    return solution_dict


def get_solution_dict(solution_file):
    solution_dict = {}
    with open(solution_file) as f:
        for line in f:
            if line[0] == "#":
                continue
            key, val = line.split()
            _, i, j, t = key.split("_")
            solution_dict[int(i), int(j), int(t)] = float(val)
    return solution_dict


def main(data, target, args):
    # Names for various stuff
    model_name = 'model_{0}_{1}_{2}.lp'.format(args.data, args.kernel, args.function)
    param_name = 'model_{0}_{1}_{2}.prm'.format(args.data, args.kernel, args.function)
    solution_name = 'solution_{0}_{1}_{2}.sol'.format(args.data, args.kernel, args.function)
    ultrametric_name = 'ultrametric_{0}_{1}_{2}'.format(args.data, args.kernel, args.function)
    var_name = 'var_{0}_{1}_{2}_{3}.pkl'.format(args.data, args.data, args.kernel, args.function)
    obj_name = 'obj_{0}_{1}_{2}_{3}.pkl'.format(args.data, args.data, args.kernel, args.function)
    laminar_name = 'laminar_{0}_{1}_{2}.pkl'.format(args.data, args.kernel, args.function)
    tree_name = 'lp_tree_{0}_{1}_{2}_{3}.pdf'.format(args.data, args.kernel, args.function, args.eps)
    err_dict_name = 'error_{0}_{1}_{2}_{3}.txt'.format(args.data, args.kernel, args.function, args.eps)


    # Test other hierarchical clustering algorithms
    one_target = map(lambda x: x + 1, target)
    k = args.prune
    y = pdist(data, metric='euclidean')
    Z = []
    Z.append(hac.linkage(y, method='single'))
    Z.append(hac.linkage(y, method='complete'))
    Z.append(hac.linkage(y, method='average'))
    ward = hac.linkage(data, method='ward')
    Z.append(ward)
    errors = []
    while Z:
        x = Z.pop(0)
        pred = hac.fcluster(x, k, 'maxclust')
        # print('pred = ', pred)
        err = utils.error(list(pred), one_target)
        errors.append(err)
    # K means
    clf = KMeans(k)
    pred = clf.fit_predict(data)
    pred = map(lambda x: x + 1, pred)
    err = utils.error(list(pred), one_target)
    errors.append(err)
    # print('kmeans = ', pred)
    error_dict = {'single linkage': errors[0], 'complete linkage': errors[1], 'average linkage': errors[2], 'ward': errors[3]}
    error_dict['kmeans'] = errors[4]
    print(error_dict)

    # initialize model
    if args.function == 'linear':
        m = init_model(data, args.kernel, args.triangle, utils.linear)
    if args.function == 'quadratic':
        m = init_model(data, args.kernel, args.triangle, utils.quadratic)
    if args.function == 'cubic':
        m = init_model(data, args.kernel, args.triangle, utils.cubic)
    if args.function == 'logarithm':
        m = init_model(data, args.kernel, args.triangle, utils.logarithm)
    if args.function == 'exponential':
        m = init_model(data, args.kernel, args.triangle, utils.exponential)
    m._n = data.shape[0]

    # Check if reading solution from file
    if args.solution:
        print('Reading LP solution from ', args.solution)
        solution_dict = read_solution(m, args.solution)
    else:
        start = time.time()
        print('Optimizing over model')
        m.optimize()
        flag = args.triangle
        while flag and time.time() - start < args.time:
            print("Time_diff = {}".format(time.time() - start))
            m.optimize()
            # Feed solution to separation oracle
            flag = separation_oracle(m, args.triangle)
        end = time.time()
        print('Total time to optimize = {0}'.format(end - start))
        print('Writing solution to ', solution_name)
        m.write(solution_name)
        print('Saving model to ', model_name)
        m.write(model_name)
        solution_dict = get_solution_dict(solution_name)

    # print('Triangle inequality satisfied: ', check_triangle_constraints(m))
    # print('Spreading constraints satisfied: ', check_spreading_constraints(m))

    # Get ultrametric from LP
    print('Rounding LP')
    if args.function == 'linear':
        d = get_ultrametric_from_lp(m, solution_dict, args.eps, utils.linear)
        utils.inverse_ultrametric(d, utils.inverse_linear)
    elif args.function == 'quadratic':
        d = get_ultrametric_from_lp(m, solution_dict, args.eps, utils.quadratic)
        utils.inverse_ultrametric(d, utils.inverse_quadratic)
    elif args.function == 'cubic':
        d = get_ultrametric_from_lp(m, solution_dict, args.eps, utils.cubic)
        utils.inverse_ultrametric(d, utils.inverse_cubic)
    elif args.function == 'exponential':
        d = get_ultrametric_from_lp(m, solution_dict, args.eps, utils.exponential)
        utils.inverse_ultrametric(d, utils.inverse_exponential)
    elif args.function == 'logarithm':
        d = get_ultrametric_from_lp(m, solution_dict, args.eps, utils.logarithm)
        utils.inverse_ultrametric(d, utils.inverse_logarithm)

    print('d = ', d)
    cost = utils.get_cost(m, d)
    print('Cost of hierarchy: ', cost)
    print('Check ultrametric: ', utils.check_ultrametric(d))
    # print(d)
    total_obj = utils.get_total(m)
    print('Total objective = ', total_obj)
    print('Scaled cost = ', cost/total_obj)

    utils.complete_ultrametric(d)
    print('Building laminar list')
    L = utils.build_laminar_list(d)
    # print('Laminar list = ', L)
    print('Check laminar: ', utils.test_laminar(L))
    labels = [1]*m._n
    pruned = utils.prune(L, one_target, k, labels)
    print('Error on pruning: ', pruned[0])
    error_dict['lp rounding'] = pruned[0]
    with open(err_dict_name, 'wb') as f:
        f.write(str(error_dict))

    # Build and draw the hierarchy
    G = utils.build_hierarchy(d)
    print('Drawing tree to ', tree_name)
    utils.draw(G, target, m._n, tree_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data file', required=True)
    parser.add_argument('-l', '--label', type=int, default=-1, help='Column of labels')
    parser.add_argument('-k', '--kernel', type=str, default='linear', help='Type of kernel')
    parser.add_argument('-t', '--triangle', action='store_true', help='lazy triangle')
    parser.add_argument('-f', '--function', type=str, default='linear', help='linear, quadratic, cubic, logarithm, exponential')
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='Number of threads')
    parser.add_argument('-e', '--eps', type=float, default=0.5, help='epsilon')
    parser.add_argument('-s', '--solution', type=str, default='', help='solution file')
    parser.add_argument('-x', '--sample', type=int, default=-1, help='sample')
    parser.add_argument('-p', '--prune', type=int, required=True, help='num of clusters in pruning')
    parser.add_argument('-ti', '--time', type=float, default=1000, help='optimize at most for this many secs')
    args = parser.parse_args()

    # prepare data
    data, target = utils.prepare_data(args.data, args.label)
    if args.sample > 0:
        data, target = utils.sample(data, target, args.sample)
    main(data, target, args)
