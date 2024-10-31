import abc
from typing import Tuple, List

from irsim.world.object_base import ObstacleInfo


class MapData:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles


class TestMetrics:
    num_of_sampled_nodes: int = 0
    num_of_collision_check: int = 0
    planning_time: float = 0
    path_length: float = 0
    reversed_path_length: float = 0
    success_rate: float = 0

    def __str__(self):
        return ''.join([
            f'number of sampled nodes: {self.num_of_sampled_nodes}\n',
            f'number of collision check: {self.num_of_collision_check}\n',
            f'planning_time: {self.planning_time}\n',
            f'path_length: {self.path_length}\n',
            f'reversed_path_length: {self.reversed_path_length}\n',
            f'success_rate: {self.success_rate}'])



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
        self.metrics: TestMetrics = TestMetrics()

    @abc.abstractmethod
    def run(self, map_data):
        pass

    @abc.abstractmethod
    def plot_path(self, save_path=None):
        pass

    def get_metrics(self):
        return self.metrics