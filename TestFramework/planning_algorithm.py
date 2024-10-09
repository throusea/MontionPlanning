import abc
from typing import Tuple, List

from irsim.world.object_base import ObstacleInfo


class MapData:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles


def importToMapData(env):
    sx, sy, syaw0, _ = env.get_robot_state()
    gx, gy, gyaw0 = env.get_robot_info().goal
    sx, sy, syaw0 = sx[0], sy[0], syaw0[0]
    gx, gy, gyaw0 = gx[0], gy[0], gyaw0[0]
    data = MapData(
        (sx, sy, syaw0),
        (gx, gy, gyaw0),
        env.get_obstacle_list()
    )
    return data


class PlanningAlgorithm(abc.ABC):
    def __init__(self, ackerman_model):
        self.c = ackerman_model

    @abc.abstractmethod
    def run(self, map_data):
        pass

    @abc.abstractmethod
    def plot_path(self):
        pass