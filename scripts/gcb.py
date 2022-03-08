#!/bin/python3

'''
 * This file is part of the Gas-Leak-Estimation.
 * *
 * @author Kalvik Jakkala
 * @contact kjakkala@uncc.edu
 * Repository: https://github.com/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation
 *
 * Copyright (C) 2020--2022 Kalvik Jakkala.
 * The Gas-Leak-Estimation repo is owned by Kalvik Jakkala and is protected by United States copyright laws and applicable international treaties and/or conventions.
 *
 * The Gas-Leak-Estimation repo is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * DISCLAIMER OF WARRANTIES: THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT. YOU BEAR ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE OR HARDWARE.
 *
 * SUPPORT AND MAINTENANCE: No support, installation, or training is provided.
 *
 * You should have received a copy of the GNU General Public License along with Gas-Leak-Estimation repo. If not, see <https://www.gnu.org/licenses/>.
'''

import ray
import subprocess
import numpy as np
from time import time
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from estimators import compute_eer, GaussEstimator, compute_leak_rate_mse

'''
Method to compute route cost
'''
def compute_dist(path, locs):
    dist = 0
    for (a, b) in path:
        dist += np.linalg.norm(locs[a]-locs[b])
    return dist

'''
Method to extract route from or-tools solution
'''
def get_routes(solution, routing, manager):
    index = routing.Start(0)
    plan_output = []
    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
    plan_output.append(manager.IndexToNode(index))
    return plan_output

'''
Method to compute tsp solution for a given problem using or-tools 
'''
@ray.remote
def run_tsp(req_nodes, distance_matrix, locs, leak_estimators, paths, max_time):
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)

    solution = None
    timeout = int(max_time-time())
    if timeout > 0:
        search_parameters.time_limit.seconds = timeout
        solution = routing.SolveWithParameters(search_parameters)

    if solution is not None:
        path = get_routes(solution, routing, manager)
        path = [req_nodes[n] for n in path]
        path_ = []
        for i in range(1, len(path)):
            edge_path = paths[path[i-1]][path[i]]
            for j in range(1, len(edge_path)):
                path_.append((edge_path[j-1], edge_path[j]))
        path = path_
        cost = compute_dist(path, locs)
        eer = compute_eer(path, locs, leak_estimators)
    else:
        path = []
        cost = 0
        eer = -np.inf
    return path, cost, eer

'''
Method to keep track of the graph and launch all the required 
TSP calls needed in each iteration of GCB
'''
class TSP():
    def __init__(self, G, locs, leak_estimators):
        self.G = G
        self.distance_matrix = np.ones((len(locs), len(locs)))
        self.paths = dict(nx.all_pairs_shortest_path(G))
        for i in range(len(locs)):
            for j in range(len(locs)):
                path = self.paths[i][j]
                path = [(path[i-1], path[i]) for i in range(1, len(path))]
                dist = np.round(compute_dist(path, locs)*100)
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
        self.distance_matrix = self.distance_matrix.astype(np.int64)
        self.locs = locs
        self.leak_estimators = leak_estimators

    def solve_problems(self, req_ids, max_time):
        path_lens = []
        results = []
        for req_id in req_ids:
            req_id = list(set(req_id))
            results.append(run_tsp.remote(req_id,
                                          self.distance_matrix[req_id][:, req_id],
                                          self.locs,
                                          self.leak_estimators,
                                          self.paths,
                                          max_time))
        result_ids = results
        done_ids = []
        while len(result_ids):
            timeout = max_time-time()
            if timeout > 0:
                done_id, result_ids = ray.wait(result_ids,
                                               timeout=float(timeout))
                done_ids.extend(done_id)
            else:
                break
        if len(done_ids) != len(req_ids):
            paths = [[]]*len(req_ids)
            path_lens = [0]*len(req_ids)
            eers = [-np.inf]*len(req_ids)
        else:
            paths, path_lens, eers = zip(*ray.get(done_ids))
        return paths, path_lens, eers

'''
Method to compute rpp solution for a given problem using line coverage library 
'''
@ray.remote
def run_arp(nodes, req_edges, non_req_edges, locs, leak_estimators, max_time):
    try:
        timeout = max_time-time()
        if timeout > 0:
            results = subprocess.run(["/users/kjakkala/coverage_ws/install/bin/slc_demo",
                                      nodes,
                                      req_edges,
                                      non_req_edges],
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True,
                                     check=True,
                                     timeout=timeout)
        else:
            raise Exception("timed out")
    except Exception as e:
        if 'timed out' not in str(e):
            print(e)
        return [], 0, -np.inf
    path = []
    for line in results.stdout.strip().split('\n')[:-1]:
        _, edge_1, edge_2, _, _, edge_cost, _ = line.strip().split(' ')
        path.append(tuple([int(edge_1), int(edge_2)]))
    path += [(path[-1][-1], path[0][0])]
    return path, compute_dist(path, locs), compute_eer(path, locs, leak_estimators)

'''
Method to keep track of the graph and launch all the required 
RPP calls needed in each iteration of GCB
'''
class ARP():
    def __init__(self, G, locs, leak_estimators):
        self.G = G
        self.nodes = ""
        for node in G.nodes():
            self.nodes += "{} 0 0\n".format(node)
        self.locs = locs
        self.leak_estimators = leak_estimators

    def gen_edge_lists(self, req_id):
        req_edges = ""
        for edge in np.array(self.G.edges())[req_id]:
            edge_cost = np.linalg.norm(self.locs[edge[0]]-self.locs[edge[1]])
            req_edges += "{0} {1} {2} {2} {2} {2}\n".format(edge[0],
                                                            edge[1],
                                                            edge_cost)
        non_req_id = list(set(range(len(self.G.edges()))).difference(req_id))
        non_req_edges = ""
        for edge in np.array(self.G.edges())[non_req_id]:
            edge_cost = np.linalg.norm(self.locs[edge[0]]-self.locs[edge[1]])
            non_req_edges += "{0} {1} {2} {2}\n".format(edge[0],
                                                        edge[1],
                                                        edge_cost)
        return req_edges, non_req_edges

    def solve_problems(self, req_ids, max_time):
        paths = []
        path_lens = []
        results = []
        for req_id in req_ids:
            req_edges, non_req_edges = self.gen_edge_lists(req_id)
            results.append(run_arp.remote(self.nodes,
                                           req_edges,
                                           non_req_edges,
                                           self.locs,
                                           self.leak_estimators,
                                           max_time))
        result_ids = list(results)
        done_ids = []
        while len(result_ids):
            timeout = max_time-time()
            if timeout > 0:
                done_id, result_ids = ray.wait(result_ids,
                                               timeout=float(timeout))
                done_ids.extend(done_id)
            else:
                break
        if len(done_ids) != len(req_ids):
            paths = [[]]*len(req_ids)
            path_lens = [0]*len(req_ids)
            eers = [-np.inf]*len(req_ids)
        else:
            paths, path_lens, eers = zip(*ray.get(results)) 
        return paths, path_lens, eers


'''
Args:
    G: networkx simple graph with N nodes
    locs: numpy array (N, 2), with locations of each graph node
    cutoff: float, max distance budget
    well_locs: numpy array (N, 2), with locations of each oil well
    leak_rates: numpy array (N,), with leak rate of each oil well
    solver: str, one of ['tsp', 'arp'] solvers
    time_limit: int, max time to iterate over paths
    opt_gcb: bool, use optimized varient of gcb
'''
def gcb(G, locs, cutoff, well_locs, leak_rates, solver='arp', time_limit=1200, opt_gcb=True):
    max_time = time()+time_limit

    leak_estimators = []
    for i in range(len(well_locs)):
        leak_estimators.append(GaussEstimator(leak_rates[i], list(well_locs[i])+[0,]))

    if solver == 'tsp':
        solver = TSP(G, locs, leak_estimators)
        E = list(G.nodes())
        S = [*list(G.edges())[0]]
    elif solver == 'arp':
        solver = ARP(G, locs, leak_estimators)
        E = list(range(len(G.edges())))
        S = [0]

    for s in S:
        E.remove(s)

    path_cost = 0
    path_eer = 0
    sol_path = []
    while len(E)>0:
        # construct required nodes/edges and get paths from solver and eers
        req_ids = [list(S+[e]) for e in E]
        paths, path_costs, eer_values = solver.solve_problems(req_ids, max_time)
        del_costs = np.array(path_costs)-path_cost
        del_costs = np.where(del_costs==0., 1e-6, del_costs)
        del_eer = np.array(eer_values)-path_eer

        # compute ratio increment and select node/edge
        x_vals = del_eer/del_costs
        x_vals = np.where(np.array(path_costs)>cutoff, -np.inf, x_vals)
        x_star_ind = np.argmax(x_vals)
        x_star = E[x_star_ind]
        if not np.isinf(np.max(x_vals)):
            S.append(x_star)
            E.remove(x_star)
            path_cost = path_costs[x_star_ind]
            path_eer = eer_values[x_star_ind]
            sol_path = paths[x_star_ind]
        elif opt_gcb:
            break

        if max_time-time() <= 0:
            break

    mse = compute_leak_rate_mse(sol_path, locs, leak_estimators)
    return sol_path, path_cost, path_eer, mse


if __name__=='__main__':
    ray.init()
    np.random.seed(0)

    p = 0.3
    num_nodes = 10
    budget = 10.
    G = nx.fast_gnp_random_graph(num_nodes, p, seed=0)
    nx.draw(G, with_labels=True)
    locs = nx.spring_layout(G, seed=0)
    well_locs = np.random.rand(num_nodes, 2)
    leak_rates = np.random.uniform(0,6,len(well_locs))

    start = time()
    path, cost, eer, mse = gcb(G, locs, budget, well_locs, leak_rates, 'tsp', 120)
    end = time()

    print("Path:", path)
    print("Cost:", cost)
    print("EER", eer)
    print("MSE:", mse)
    print("Time:", end-start)
