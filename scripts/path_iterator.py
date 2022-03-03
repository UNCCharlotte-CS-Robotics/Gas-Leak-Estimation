#!/bin/python3

import ray
import psutil
import numpy as np
import networkx as nx
from time import time
from estimators import compute_eer, GaussEstimator, compute_leak_rate_mse


def node_to_edge_list(path):
    path_ = []
    for i in range(len(path)-1):
        path_.append((path[i], path[i+1]))
    return path_

def compute_dist(path, locs):
    dist = 0
    for i in range(1, len(path)):
        dist += np.linalg.norm(locs[path[i-1]]-locs[path[i]])
    return dist

def available_resources():
    res = ray.available_resources()
    if 'CPU' in res.keys():
        return res['CPU']
    else:
        return 0

def chunk_lists(seq, size):
    return (seq[i::size] for i in range(size))

@ray.remote
def get_paths(G, stack, target, cutoff, locs, leak_estimators, start_time, time_limit):
    start = time()
    cpu_thresh = psutil.cpu_count()/4
    sols = []
    while len(stack):
        child, visited = stack.pop()
        if compute_dist(visited, locs) <= cutoff:
            if child == target:
                path = node_to_edge_list(visited + [target])
                eer = compute_eer(path, locs, leak_estimators)
                sols.append((path, eer))
            visited.append(child)
            for node in list(G[child].keys()):
                stack.append((node, list(visited)))
        ctime = time()
        if available_resources()>cpu_thresh and ctime-start>60:
            break
        elif (ctime-start)%60==0 and len(sols)>0:
            paths, eer_values = zip(*sols)
            eer = np.max(eer_values)
            path = paths[np.argmax(eer_values)]
            sols = [(path, eer)]
        elif ctime-start_time > time_limit:
            break
    return sols, stack

'''
Args:
    G: networkx simple graph with N nodes
    locs: numpy array (N, 2), with locations of each graph node
    cutoff: float, max distance budget
    well_locs: numpy array (N, 2), with locations of each oil well
    leak_rates: numpy array (N,), with leak rate of each oil well
    time_limit: int, max time to iterate over paths
'''
def path_iterator(G, locs, cutoff, well_locs, leak_rates, time_limit=1200):
    leak_estimators = []
    for i in range(len(well_locs)):
        leak_estimators.append(GaussEstimator(leak_rates[i], list(well_locs[i])+[0,]))

    # Get initial set of paths and search tree on single node
    start_time = time()
    start = list(G.edges())[0][0]
    step = list(G.edges())[0][1]
    stack = []
    for node in list(G[step].keys()):
        stack.append((node, list([start, step])))
    futures = get_paths.remote(G, stack, start, cutoff, locs, leak_estimators, start_time, time_limit)
    sols, stack = ray.get(futures)

    # Distribute search tree across all cpus
    while stack and time()-start_time < time_limit:
        stack = chunk_lists(stack, psutil.cpu_count())
        futures = [get_paths.remote(G, s, start, cutoff, locs, leak_estimators, start_time, time_limit) for s in stack]
        futures = ray.get(futures)
        future_sols, future_stacks = zip(*futures)
        stack = []
        for stack_ in future_stacks:
            stack.extend(stack_)
        for sol in future_sols:
            sols.extend(sol)

    if len(sols)==0:
        mse = compute_leak_rate_mse([], locs, leak_estimators)
        return [], 0, 0, mse

    paths, eer_values = zip(*sols)
    eer = np.max(eer_values)
    path = paths[np.argmax(eer_values)]
    cost = compute_dist([a for (a, b) in path], locs)
    mse = compute_leak_rate_mse(path, locs, leak_estimators)
    return path, cost, eer, mse


if __name__=='__main__':
    ray.init()
    np.random.seed(0)

    p = 0.2
    num_nodes = 20
    budget = 10.
    G = nx.fast_gnp_random_graph(num_nodes, p, seed=0)
    nx.draw(G, with_labels=True)
    locs = nx.spring_layout(G, seed=0)
    well_locs = np.random.rand(num_nodes, 2)
    leak_rates = np.random.uniform(0,6,len(well_locs))

    start = time()
    path, cost, eer, mse = path_iterator(G, locs, budget, well_locs, leak_rates, 60)
    end = time()

    print("Path:", path)
    print("Cost:", cost)
    print("EER", eer)
    print("MSE:", mse)
    print("Time:", end-start)
