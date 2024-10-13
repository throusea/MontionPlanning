import abc
import math
import os
import re
import sys
import time
from typing import Tuple, List

import numpy as np
import pathlib
from irsim import EnvBase
from matplotlib import pyplot as plt

from planning_algorithm import importToMapData
import hybrid_astar
import rrt_star_reeds_shepp
import model_const
sys.path.append(str(pathlib.Path(__file__).parent.parent))

FIG_INDEX = 1

def get_map_data(map_path):
    env = EnvBase(map_path, save_ani=False, display=True, full=False)
    return importToMapData(env)


def test_by_one_episode(map_path, planning_algorithm):
    global FIG_INDEX
    map_data = get_map_data(map_path)

    t0 = time.time()
    path = planning_algorithm.run(map_data)
    t1 = time.time()
    print("running T: ", t1 - t0)

    if not path:
        print("Searching failed!")

    planning_algorithm.plot_path(f'../../examples/figures/{FIG_INDEX}s.png')
    FIG_INDEX += 1
    test_metrics = planning_algorithm.get_metrics()
    return test_metrics


if __name__ == '__main__':
    algorithm = hybrid_astar.HybridAstar(model_const.C)
    # algorithm = rrt_star_reeds_shepp.MyRRTStarReedsShepp(model_const.C)
    map_data_dir = '../../examples/map_data'

    for file in os.listdir(map_data_dir):
        file_path = os.path.join(map_data_dir, file)
        if (os.path.isfile(file_path) and
            re.match(r'.*m7t2e6.*', file_path)):
            # re.match(r'.*m\d.yaml', file_path)):
            test_metrics = test_by_one_episode(file_path, algorithm)
            print(test_metrics)
        if FIG_INDEX >= 10:
            break