from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
from scipy.spatial.distance import cosine
import time
import cPickle as pickle
import math
from gurobipy import *
from cluster import error
import utils


def nontrivial_constraints(m):
    # Constraints to make ultrametric non-trivial
    n = m._n
    # Get variables
    flag_spreading = False
    flag_hereditary = False
    for t in xrange(1, n):
        visited = [0]*n
        for i in xrange(1, n):
            # t iterates over the sizes of the set S
            pre_var_list = [m.getVarByName('x_' + str(j) + '_' + str(i) + '_' + str(t)) for j in xrange(i)]
            post_var_list = [m.getVarByName('x_' + str(i) + '_' + str(j) + '_' + str(t)) for j in xrange(i + 1, n)]
            pre_var_sol = map(lambda x: int(round(x)), m.cbGetSolution(pre_var_list))
            post_var_sol = map(lambda x: int(round(x)), m.cbGetSolution(post_var_list))
            # count + 1 = |S| in the spreading constraint, \sum x^t_ij \ge count - t
            pre_zero_set = set([j for j in xrange(i) if pre_var_sol[j] == 0])
            post_zero_set = set([j for j in xrange(i + 1, n) if post_var_sol[j - i - 1] == 0])
            count = len(pre_zero_set) + len(post_zero_set)
            # Delete the lists
            del pre_var_list
            del post_var_list
            del pre_var_sol
            del post_var_sol
            # Spreading constraint
            flag_spreading = spreading_constraint(m, pre_zero_set, post_zero_set, i, t)
            # Hereditary constraint
            flag_hereditary = hereditary_constraint(m, pre_zero_set, post_zero_set, i, t, visited)
            # Delete the sets
            del pre_zero_set
            del post_zero_set
    return flag_spreading or flag_hereditary


def hereditary_constraint(m, pre_zero_set, post_zero_set, i, t, visited):
    # Apply hereditary constraint
    count = len(pre_zero_set) + len(post_zero_set)
    h_constr_name = 'hereditary.constr'
    flag = False
    if not visited[i] and count + 1 < t:
        visited[i] = 1
        for j in pre_zero_set:
            visited[j] = 1
            vname = 'x_' + str(j) + '_' + str(i) + '_' + str(count + 1)
            v = m.getVarByName('x_' + str(j) + '_' + str(i) + '_' + str(count + 1))
            val = m.cbGetSolution(v)
            vtname = 'x_' + str(j) + '_' + str(i) + '_' + str(t)
            vt = m.getVarByName(vtname)
            if val:
                with open(h_constr_name, 'a') as f:
                    f.write(vname + ',' + vtname + '\n')
                m.cbLazy(v <= vt)
                flag = True
    return flag


def add_hereditary_constraints(m):
    # Add hereditary constraint from file for debugging purposes
    h_constr_name = 'hereditary.constr'
    with open(h_constr_name, 'rb') as f:
       constraints = [x.strip('\n').split(',') for x in f.readlines()]
    for c in constraints:
        cname = 'h_constr_' + c[0] + '_' + c[1]
        v = m.getVarByName(c[0])
        vt = m.getVarByName(c[1])
        m.addConstr(v <= vt, name=cname)
    m.update()


def spreading_constraint(m, pre_zero_set, post_zero_set, i, t):
    # Apply spreading constraint
    count = len(pre_zero_set) + len(post_zero_set)
    s_constr_name = 'spreading.constr'
    flag = False
    if count >= t:
        expr = 0
        cname = str()
        for j in pre_zero_set:
            vname = 'x_' + str(j) + '_' + str(i) + '_' + str(t)
            expr += m.getVarByName(vname)
            cname += vname + ','
        for j in post_zero_set:
            vname = 'x_' + str(i) + '_' + str(j) + '_' + str(t)
            expr += m.getVarByName(vname)
            cname += vname + ','
        m.cbLazy(expr >= count + 1 - t)
        cname += str(count + 1 - t)
        with open(s_constr_name, 'a') as f:
            f.write(cname + '\n')
        flag = True
    return flag


def add_spreading_constraints(m):
    # Add spreading constraints from file for debugging purposes
    s_constr_name = 'spreading.constr'
    with open(s_constr_name, 'rb') as f:
        constraints = [x.strip('\n').split(',') for x in f.readlines()]
    for c in constraints:
        expr = 0
        for i in xrange(len(c) - 1):
            expr += m.getVarByName(c[i])
        m.addConstr(expr >= int(c.pop()))
    m.update()


def triangle_constraints(m):
    # Lazy triangle inequality
    n = m._n
    d = {}
    flag = False
    for i in xrange(n):
        for j in xrange(i + 1, n):
            for t in xrange(1, n):
                v_ijt = m.getVarByName('x_' + str(i) + '_' + str(j) + '_' + str(t))
                x_ijt = m.cbGetSolution(v_ijt)
                for k in xrange(j + 1, n):
                    v_ikt = m.getVarByName('x_' + str(i) + '_' + str(k) + '_' + str(t))
                    v_jkt = m.getVarByName('x_' + str(j) + '_' + str(k) + '_' + str(t))
                    x_ikt = m.cbGetSolution(v_ikt)
                    x_jkt = m.cbGetSolution(v_jkt)
                    if int(round(x_ijt)) > int(round(x_ikt)) + int(round(x_jkt)):
                        flag = True
                        m.cbLazy(v_ijt <= v_ikt + v_jkt)
                    if int(round(x_ikt)) > int(round(x_ijt)) + int(round(x_jkt)):
                        flag = True
                        m.cbLazy(v_ikt <= v_ijt + v_jkt)
                    if int(round(x_jkt)) > int(round(x_ijt)) + int(round(x_ikt)):
                        flag = True
                        m.cbLazy(v_jkt <= v_ijt + v_ikt)
    return flag


def callback_function(m, where):
    # First cut off triangle, then spreading
    if where == GRB.Callback.MIPSOL:
        triangle_constraints(m)
        nontrivial_constraints(m)


def add_variables(data, similarity, target, m, f):
    # Add variables to m
    print('Adding variables to model')
    v = {}
    w = {}
    start = time.time()
    counter = 0
    n = m._n
    initial_solution = {}
    for i in range(n):
        for j in range(i + 1, n):
            for t in range(1, n):
                # w[i, j, t] is the kernel function
                w[i, j, t] = (f(t) - f(t-1)) * similarity(i, j)
                vname = 'x_' + str(i) + '_' + str(j) + '_' + str(t)
                v[i, j, t] = m.addVar(lb=0.0, ub=1.0, obj=w[i, j, t], vtype=GRB.BINARY, name=vname)
                counter += 1
                if target[i] == target[j]:
                    if t <= 9:
                        initial_solution[i, j, t] = 1
                    else:
                        initial_solution[i, j, t] = 1
                else:
                    if t <= 29:
                        initial_solution[i, j, t] = 1
                    else:
                        initial_solution[i, j, t] = 1
    m.update()
    for i in range(n):
        for j in range(i + 1, n):
            for t in range(1, n):
                v[i, j, t].start = initial_solution[i, j, t]
    m.update()
    m._vars = v
    m._obj = w
    print('Adding initial starting solution')
    end = time.time()
    print('Time to add variables = {0:.2f}s'.format(end - start))


def add_layer_constraints(m):
    # Add layer constraints serially
    print('Adding layer constraints')
    start = time.time()
    n = m._n
    for t in range(1, n - 1):
        for i in range(n):
            for j in range(i + 1, n):
                cname = 'layer_' + str(i) + '_' + str(j) + '_' + str(t)
                m.addConstr(m._vars[i, j, t] - m._vars[i, j, t + 1] >= 0, name=cname)
    m.update()
    end = time.time()
    print('Time to add layer inequalities = {0:2f}s'.format(end - start))


def init_model(data, target, kernel, f):
    # Initiliaze Gurobi model
    m = Model()
    m._n = data.shape[0]
    # Add variables
    add_variables(data, target, m, kernel, f)
    # Add layer constraints
    add_layer_constraints(m)
    # Add constraints from file
    # add_spreading_constraints(m)
    # add_hereditary_constraints(m)
    m.modelSense = GRB.MINIMIZE
    m.params.LazyConstraints = 1
    return m


def get_ultrametric(m, f):
    # Recover ultrametric from binary solution
    d = {}
    n = m._n
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = 0
            for t in range(1, n):
                v = m.getVarByName('x_{0}_{1}_{2}'.format(i, j, t))
                d[i, j] += (f(t) - f(t-1)) * int(round(v.X))
    return d


def check_binary_triangle(m):
    # Check if triangle inequality is satisfied by every binary solution
    n = m._n
    for t in range(1, n):
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    v_ij = m.getVarByName('x_' + str(i) + '_' + str(j) + '_' + str(t))
                    v_jk = m.getVarByName('x_' + str(j) + '_' + str(k) + '_' + str(t))
                    v_ik = m.getVarByName('x_' + str(i) + '_' + str(k) + '_' + str(t))
                    if int(round(v_ik.X)) > max(int(round(v_ij.X)), int(round(v_jk.X))):
                        print('i = {0}, j = {1}, k = {2}, t = {3}'.format(i, j, k, t))
                        print('v_ij = {0:2f}, v_jk = {1:2f}. v_ik = {2:2f}'.format(v_ij.X, v_jk.X, v_ik.X))
                        return False
                    if int(round(v_ij.X)) > max(int(round(v_ik.X)), int(round(v_jk.X))):
                        print('i = {0}, j = {1}, k = {2}, t = {3}'.format(i, j, k, t))
                        print('v_ij = {0:2f}, v_jk = {1:2f}. v_ik = {2:2f}'.format(v_ij.X, v_jk.X, v_ik.X))
                        return False
                    if int(round(v_jk.X)) > max(int(round(v_ij.X)), int(round(v_ik.X))):
                        print('i = {0}, j = {1}, k = {2}, t = {3}'.format(i, j, k, t))
                        print('v_ij = {0:2f}, v_jk = {1:2f}. v_ik = {2:2f}'.format(v_ij.X, v_jk.X, v_ik.X))
                        return False
    return True


def main(data, target, args):
    model_name = 'model_{0}_{1}_{2}.lp'.format(args.data, args.kernel, args.function)
    param_name = 'model_{0}_{1}_{2}.prm'.format(args.data, args.kernel, args.function)
    solution_name = 'solution_{0}_{1}_{2}.sol'.format(args.data, args.kernel, args.function)
    ultrametric_name = 'ultrametric_{0}_{1}_{2}'.format(args.data, args.kernel, args.function)
    var_name = 'var_{0}_{1}_{2}_{3}.pkl'.format(args.data, args.data, args.kernel, args.function)
    obj_name = 'obj_{0}_{1}_{2}_{3}.pkl'.format(args.data, args.data, args.kernel, args.function)
    laminar_name = 'laminar_{0}_{1}_{2}.pkl'.format(args.data, args.kernel, args.function)
    tree_name = 'ip_tree_{0}_{1}_{2}.pdf'.format(args.data, args.kernel, args.function)
    if args.kernel == 'cosine':
        y = pdist(data, metric='cosine')
        # Make condensed distance matrix into redundant form
        similarity = 1 - y
        similarity = squareform(similarity)
    if args.kernel == 'gaussian':
        y = pdist(data, metric='sqeuclidean')
        s = 1
        y = 1 - np.exp(-(y**2)/(2*s ** 2))
        # Make condensed distance matrix into redundant form
        similarity = 1 - y
        similarity = squareform(similarity)
    if args.kernel == 'sqeuclidean':
        y = pdist(data, metric='sqeuclidean')
        similarity = - y
        similarity = squareform(similarity)
    if args.function == 'linear':
        m = init_model(data, similarity, target, utils.linear)
    elif args.function == 'quadratic':
        m = init_model(data, similarity, target, utils.quadratic)
    elif args.function == 'cubic':
        m = init_model(data, similarity, target, utils.cubic)
    elif args.function == 'exponential':
        m = init_model(data, similarity, target, utils.exponential)
    elif args.function == 'logarithm':
        m = init_model(data, similarity, target, utils.logarithm)
    else:
        exit(0)
    print('Saving model')
    m.write(model_name)
    # Use concurrent optimization
    m.params.method = 3
    # Limit memory
    m.params.NodeFileStart = 10
    # Limit number of threads
    m.params.Threads = args.num_threads
    # Set MIP Focus
    m.params.MIPFocus = 3
    # Tune parameters
    print('Tuning parameters')
    m.params.tuneResults = 1
    m.tune()
    if m.tuneResultCount > 0:
        m.getTuneResult(0)
    # Set MIP Gap
    m.params.MIPGap = 0.01
    print('Saving model parameters')
    m.write(param_name)
    print('Saving objective functions')
    with open(obj_name, 'wb') as f:
        pickle.dump(m._obj, f)
    print('Optimizing over model')
    m._n = data.shape[0]
    m.optimize(callback_function)
    if m.status == GRB.Status.OPTIMAL:
        # Write solution
        m.write(solution_name)
        print('Check binary triangle for solution: ', check_binary_triangle(m))

        # Get ultrametric
        if args.function == 'linear':
            d = get_ultrametric(m, utils.linear)
            utils.inverse_ultrametric(d, utils.inverse_linear)
        elif args.function == 'quadratic':
            d = get_ultrametric(m, utils.quadratic)
            utils.inverse_ultrametric(d, utils.inverse_quadratic)
        elif args.function == 'cubic':
            d = get_ultrametric(m, utils.cubic)
            utils.inverse_ultrametric(d, utils.inverse_cubic)
        elif args.function == 'exponential':
            d = get_ultrametric(m, utils.exponential)
            utils.inverse_ultrametric(d, utils.inverse_exponential)
        elif args.function == 'logarithm':
            d = get_ultrametric(m, utils.logarithm)
            utils.inverse_ultrametric(d, utils.inverse_logarithm)


        print('d = ', d)
        print('Check ultrametric: ', utils.check_ultrametric(d))
        cost = utils.get_cost(m, d)
        print('Cost of hierarchy = ', cost)
        total_obj = utils.get_total(m)
        print('Total cost = ', total_obj)
        print('Scaled cost = ', cost/total_obj)


        # Complete ultrametric
        utils.complete_ultrametric(d)

        # Build laminar list from d
        print('building laminar list')
        L = utils.build_laminar_list(d)
        print('L = ', L)
        print('Check laminar: ', utils.test_laminar(L))
        labels = [1]*m._n
        one_target = map(lambda x: x + 1, target)

        # Prune laminar list
        pruned = utils.prune(L, one_target, args.prune, labels)
        print('Error on pruning: ', pruned[0])
        with open(ultrametric_name, 'wb') as f:
            pickle.dump(d, f)

        # Build hierarchy
        print('Building hierarchy')
        G = utils.build_hierarchy(d)

        # Draw hierarchy
        print('Drawing hierarchy to ', tree_name)
        utils.draw(G, target, m._n, tree_name)
    elif m.status == GRB.Status.INFEASIBLE:
        # Compute IIS, for debugging purposes
        m.computeIIS()
        m.write('infeasible.ilp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data file', required=True)
    parser.add_argument('-l', '--label', type=int, default=0, help='Column of labels')
    parser.add_argument('-k', '--kernel', type=str, default='linear', help='Type of kernel')
    parser.add_argument('-f', '--function', type=str, default='linear', help='linear, quadratic, cubic, exponential, logarithm')
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='Number of threads')
    parser.add_argument('-p', '--prune', type=int, required=True, help='Number of flat clusters')
    parser.add_argument('-x', '--sample', type=int, default=-1, help='Num samples')
    args = parser.parse_args()
    data, target = utils.prepare_data(args.data, args.label)
    if args.sample > 0:
        data, target = utils.sample(data, target, args.sample)
    main(data, target, args)
