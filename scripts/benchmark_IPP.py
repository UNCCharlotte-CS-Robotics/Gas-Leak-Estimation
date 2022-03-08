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

import os
import ray
import sys
import time
import json
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict

from gcb import gcb
from path_iterator import path_iterator


def compute_dist(G, locs):
    path = list(G.edges())
    dist = 0
    for (a, b) in path:
        dist += np.linalg.norm(locs[a]-locs[b])
    return dist

if __name__=='__main__':
    ray.init()
    np.random.seed(0)

    st = int(sys.argv[1])
    end = st+16
    timeout = 60*20 # 20 mins

    filename = 'benchmark_gcb_iter-{}_{}.json'.format(st, end)
    map_folder = "../data/"
    maps = []
    for f in os.listdir(map_folder):
        if os.path.isfile(os.path.join(map_folder, f)):
            maps.append(os.path.join(map_folder, f))
    maps.sort()

    data = defaultdict(lambda: defaultdict(int))
    for map in maps[st:end]:
        with open(map, 'rb') as file:
            data_dict = pickle.load(file)

        map = map.split('/')[-1].split('.')[0]

        G = data_dict['graph']
        locs = data_dict['locs']
        well_locs = data_dict['wells']
        norm = data_dict['norm']
        dist_budget = 15000.0/norm
        leak_rates = np.random.uniform(0,6,len(well_locs))
        data['dist_budget'] = dist_budget
        data['leak_rates'] = leak_rates.tolist()

        print("\nMap:", map)
        print("# Nodes:", len(list(G.nodes())), flush=True)
        print("# Edges:", len(list(G.edges())), flush=True)
        print("# Budget: {:.4f}".format(dist_budget), flush=True)
        print("Total Distance: {:.4f}".format(compute_dist(G, locs)), flush=True)

        start_time = time.time()
        path, cost, eer, mse = path_iterator(G, locs, dist_budget, well_locs, leak_rates, timeout)
        end_time = time.time()
        data['iter_time'][map] = end_time-start_time
        data['iter_eer'][map] = eer
        data['iter_path'][map] = path
        data['iter_cost'][map] = cost
        data['iter_mse'][map] = mse

        start_time = time.time()
        path, cost, eer, mse = gcb(G, locs, dist_budget, well_locs, leak_rates, 'arp', timeout)
        end_time = time.time()
        data['arp_gcb_time'][map] = end_time-start_time
        data['arp_gcb_eer'][map] = eer
        data['arp_gcb_path'][map] = path
        data['arp_gcb_cost'][map] = cost
        data['arp_gcb_mse'][map] = mse

        start_time = time.time()
        path, cost, eer, mse = gcb(G, locs, dist_budget, well_locs, leak_rates, 'arp', timeout, False)
        end_time = time.time()
        data['varp_gcb_time'][map] = end_time-start_time
        data['varp_gcb_eer'][map] = eer
        data['varp_gcb_path'][map] = path
        data['varp_gcb_cost'][map] = cost
        data['varp_gcb_mse'][map] = mse

        start_time = time.time()
        path, cost, eer, mse = gcb(G, locs, dist_budget, well_locs, leak_rates, 'tsp', timeout)
        end_time = time.time()
        data['tsp_gcb_time'][map] = end_time-start_time
        data['tsp_gcb_eer'][map] = eer
        data['tsp_gcb_path'][map] = path
        data['tsp_gcb_cost'][map] = cost
        data['tsp_gcb_mse'][map] = mse

        start_time = time.time()
        path, cost, eer, mse = gcb(G, locs, dist_budget, well_locs, leak_rates, 'tsp', timeout, False)
        end_time = time.time()
        data['vtsp_gcb_time'][map] = end_time-start_time
        data['vtsp_gcb_eer'][map] = eer
        data['vtsp_gcb_path'][map] = path
        data['vtsp_gcb_cost'][map] = cost
        data['vtsp_gcb_mse'][map] = mse

        print("EER")
        print("Iter: {:.4f}, CGCB: {:.4f}, VCGCB: {:.4f}, TGCB: {:.4f}".format(data['iter_eer'][map],
                                                               data['arp_gcb_eer'][map],
                                                               data['varp_gcb_eer'][map],
                                                               data['tsp_gcb_eer'][map]))
        print("Time")
        print("Iter: {:.4f}, CGCB: {:.4f}, VCGCB: {:.4f}, TGCB: {:.4f}".format(data['iter_time'][map],
                                                               data['arp_gcb_time'][map],
                                                               data['varp_gcb_time'][map],
                                                               data['tsp_gcb_time'][map]),
                                                               flush=True)
        print("MSE")
        print("Iter: {:.4f}, CGCB: {:.4f}, VCGCB: {:.4f}, TGCB: {:.4f}".format(data['iter_mse'][map],
                                                               data['arp_gcb_mse'][map],
                                                               data['varp_gcb_mse'][map],
                                                               data['tsp_gcb_mse'][map]),
                                                               flush=True)

        json.dump(data, open(filename, 'w'))

    print("\nAvg EER")
    print("Iter: {:.4f}, CGCB: {:.4f}, VCGCB: {:.4f}, TGCB: {:.4f}".format(np.mean(list(data['iter_eer'].values())),
                                                           np.mean(list(data['arp_gcb_eer'].values())),
                                                           np.mean(list(data['varp_gcb_eer'].values())),
                                                           np.mean(list(data['tsp_gcb_eer'].values()))))
    print("Avg Time")
    print("Iter: {:.4f}, CGCB: {:.4f}, VCGCB: {:.4f}, TGCB: {:.4f}".format(np.mean(list(data['iter_time'].values())),
                                                           np.mean(list(data['arp_gcb_time'].values())),
                                                           np.mean(list(data['varp_gcb_time'].values())),
                                                           np.mean(list(data['tsp_gcb_time'].values()))),
                                                           flush=True)
