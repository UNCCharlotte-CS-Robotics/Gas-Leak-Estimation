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
import pickle
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
from geopandas import GeoDataFrame
from shapely.geometry import Point
from sklearn.cluster import KMeans


'''
Method to compute route cost
'''
def compute_dist(G, locs):
    path = list(G.edges())
    dist = 0
    for (a, b) in path:
        dist += np.linalg.norm(locs[a]-locs[b])
    return dist


'''
Extract road network graph using osmnx from region around given oil wells
'''
@ray.remote
def get_graph(well_locs, data_filename):
    # Get map from coordinates
    lr = np.max(well_locs, axis=0)
    ul = np.min(well_locs, axis=0)
    try:
        G_map = ox.graph_from_bbox(lr[0], ul[0], lr[1], ul[1],
                                   network_type='drive_service')
    except:
        return None

    # project coordinates and relabel nodes
    G_map = ox.project_graph(G_map)
    mapping = {node: i for i, node in enumerate(G_map.nodes())}
    G_map = nx.relabel_nodes(G_map, mapping)

    # save node coordinates to dict
    nodes = ox.graph_to_gdfs(G_map, edges=False)[['x', 'y']].values
    locs={}
    for i, node in enumerate(G_map.nodes()):
        locs[node] = (nodes[i])

    # convert graph to simple graph
    G = nx.Graph()
    G.add_edges_from(G_map.edges())
    G.remove_edges_from(nx.selfloop_edges(G))

    if not (len(well_locs) > 10 and len(G.nodes()) > 10):
        return None

    # project well coordinates
    gdf = GeoDataFrame({'geometry': [Point(well[1], well[0]) for well in well_locs]},
                       crs='epsg:4326')
    gdf = gdf.to_crs(ox.graph_to_gdfs(G_map, edges=False).crs)
    well_locs = []
    for pt in gdf.values:
        well_locs.append([pt[0].x, pt[0].y])
    well_locs = np.array(well_locs)

    # Standardize all coordinates
    center = np.min([np.array(list(locs.values())).min(axis=0),
                     well_locs.min(axis=0)], axis=0)
    locs = dict(zip(locs.keys(), np.array(list(locs.values()))-center))
    well_locs -= center

    norm = np.max([np.array(list(locs.values())).max(),
                   well_locs.max()])
    locs = dict(zip(locs.keys(), np.array(list(locs.values()))/norm))
    well_locs /= norm

    dist_budget = 5*1609.34/norm

    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if dist_budget*2 < compute_dist(G, locs) and nx.average_node_connectivity(G) < 1.1:
        data = {'graph': G,
                'locs': locs,
                'wells': well_locs,
                'lr': lr,
                'ul': ul,
                'norm': norm}

        with open(data_filename, 'wb') as file:
            pickle.dump(data, file)


if __name__=='__main__':
    ray.init()

    well_data_filepath = "../US_WellsFacilities2016/TX_OGfacil_2016.csv"
    dataset_path = "../data/"
    data_filepath = os.path.join(dataset_path, "map_{}.pkl")
    os.mkdir(dataset_path)

    wells = pd.read_csv(well_data_filepath, delimiter=',')
    wells = wells[['LATITUDE', 'LONGITUDE']].values
    wells = wells[:int(len(wells)/5)]
    num_clusters = int(len(wells)/40)
    kmeans = KMeans(n_clusters=num_clusters).fit(wells)

    futures = [get_graph.remote(wells[np.where(kmeans.labels_==i)[0]], data_filepath.format(str(i))) for i in range(num_clusters)]
    ray.get(futures)
