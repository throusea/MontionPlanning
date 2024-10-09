import abc
import math
import sys
import time
from typing import Tuple, List

import numpy as np
import pathlib
from irsim import EnvBase
from matplotlib import pyplot as plt

from planning_algorithm import importToMapData
import hybrid_astar
import model_const
sys.path.append(str(pathlib.Path(__file__).parent.parent))


class TestMetrics:
    num_of_sampled_nodes: int = 0
    num_of_collision: int = 0
    planning_time: float = 0
    length_of_path: float = 0
    length_of_reversed_path: float = 0
    success_rate: float = 0


def get_map_data(map_path):
    env = EnvBase(map_path, save_ani=False, display=True, full=False)
    return importToMapData(env)


def test_by_one_episode(map_path, planning_algorithm):
    map_data = get_map_data(map_path)

    t0 = time.time()
    path = planning_algorithm.run(map_data)
    t1 = time.time()
    print("running T: ", t1 - t0)

    if not path:
        print("Searching failed!")
    else:
        planning_algorithm.plot_path()
    test_metrics = TestMetrics()
    return test_metrics


if __name__ == '__main__':
    algorithm = hybrid_astar.HybridAstar(model_const.C)
    test_by_one_episode('..\\..\\ex.yaml', algorithm)
