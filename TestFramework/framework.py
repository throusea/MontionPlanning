import abc
import math
import os
import re
import sys
import time
from typing import Tuple, List

import numpy as np
import pathlib

import pandas as pd
from irsim import EnvBase
from matplotlib import pyplot as plt

from planning_algorithm import importToMapData, TestMetrics
import hybrid_astar
import rrt_star_reeds_shepp
import model_const
sys.path.append(str(pathlib.Path(__file__).parent.parent))

def get_map_data(map_path):
    env = EnvBase(map_path, save_ani=False, display=True, full=False)
    return importToMapData(env)


def test_by_one_episode(map_path, algorithm_name, map_name, garage_name, garage_index, planning_algorithm):
    map_data = get_map_data(map_path)

    t0 = time.time()
    path = planning_algorithm.run(map_data)
    t1 = time.time()
    print("running T: ", t1 - t0)

    figure_path = f'../../examples/figures/{algorithm_name}/{map_name}/{garage_name}/{garage_index}.png'
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    planning_algorithm.plot_path(figure_path)
    test_metrics = planning_algorithm.get_metrics()
    return test_metrics


def add_metrics(data, name, metrics: TestMetrics):
    data['name'].append(name)
    data['number of sampled nodes'].append(metrics.num_of_sampled_nodes)
    data['number of collision check'].append(metrics.num_of_collision_check)
    data['planning time'].append(metrics.planning_time)
    data['path length'].append(metrics.path_length)
    data['reversed path length'].append(metrics.reversed_path_length)
    data['success rate'].append(metrics.success_rate)

def create_data():
    data = {
        'name': [],
        'number of sampled nodes': [],
        'number of collision check': [],
        'planning time': [],
        'path length': [],
        'reversed path length': [],
        'success rate': []
    }
    return data

def main():
    # configuration
    map_name = 'm7t2e6r2'
    algorithm_name = 'hybrid_astar'

    if algorithm_name == 'hybrid_astar':
        algorithm = hybrid_astar.HybridAstar(model_const.C)
    else:
        algorithm = rrt_star_reeds_shepp.MyRRTStarReedsShepp(model_const.C)
    map_data_dir = f'../../examples/map_data/{algorithm_name}/{map_name}'
    if not os.path.exists(map_data_dir):
        os.makedirs(map_data_dir)

    for file in os.listdir(map_data_dir):
        garage_path = os.path.join(map_data_dir, file)

        # if (os.path.isfile(file_path) and
        #     re.match(r'.*m7t2e6.*', file_path)):
        #     # re.match(r'.*m\d.yaml', file_path)):
        #     test_metrics = test_by_one_episode(file_path, algorithm)
        #     print(test_metrics)
        if os.path.isdir(garage_path):
            data = create_data()
            garage_name = os.path.basename(garage_path)
            for p in os.listdir(garage_path):
                file_path = os.path.join(garage_path, p)
                if os.path.isfile(file_path):
                    garage_index = os.path.basename(file_path).split('.')[0]
                    test_metrics = test_by_one_episode(file_path, algorithm_name, map_name, garage_name, garage_index, algorithm)
                    add_metrics(data, garage_name, test_metrics)
            df = pd.DataFrame(data)
            csv_path = f'../../examples/test_data/{algorithm_name}/{map_name}/{garage_name}.csv'
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path)

if __name__ == '__main__':
    main()
