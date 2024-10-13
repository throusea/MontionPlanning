"""

Path planning Sample Code with RRT with Reeds-Shepp path

author: AtsushiSakai(@Atsushi_twi)

"""
import copy
import math
import random
import sys
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from irsim.world.object_base import ObstacleInfo
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from RRTStarReedsShepp.rrt_star_reeds_shepp import RRTStarReedsShepp
from RRT.rrt import RRT
from planning_algorithm import PlanningAlgorithm, MapData

show_animation = True


class MyRRTStarReedsShepp(PlanningAlgorithm):

    def __init__(self, ackerman_model):
        super().__init__(ackerman_model)
        self.rrt = None
        self.path = None
        self.ox, self.oy = [], []

    def run(self, map_data: MapData):
        start, goal, obstacle_list = map_data.start, map_data.goal, map_data.obstacles
        obstacle_list = self.get_local_obstacles(obstacle_list)
        self.rrt = RRT(start, goal, obstacle_list, [0, 80], max_iter=100)
        self.path = self.rrt.planning(animation=True)


    def get_local_obstacles(self, obstacles: List[ObstacleInfo]):
        ox, oy = [], []
        for obs in obstacles:
            if obs.cone_type == 'Rpositive':
                # print(obs.vertex)
                for i in range(len(obs.vertex[0])):
                    p1 = Point(obs.vertex[0][i - 1], obs.vertex[1][i - 1])
                    p2 = Point(obs.vertex[0][i], obs.vertex[1][i])
                    line = LineString([p1, p2])
                    num_points = 5
                    ratios = np.linspace(0, 1, num_points)
                    discrete_points = [line.interpolate(ratio, normalized=True) for ratio in ratios]
                    ox.extend([p.x for p in discrete_points])
                    oy.extend([p.y for p in discrete_points])
            else:
                vx = [obs.vertex[0][i] for i in range(0, len(obs.vertex[0]), 8)]
                vy = [obs.vertex[1][i] for i in range(0, len(obs.vertex[1]), 8)]
                ox.extend(vx)
                oy.extend(vy)
        self.ox, self.oy = ox, oy
        new_obstacles = zip(ox, oy, [0.2] * len(ox))
        return list(new_obstacles)


    def plot_path(self, save_path=None):
        x_list = self.path.x
        y_list = self.path.y
        yaw_list = self.path.yaw
        direction = self.path.direction
        np.save('..\\..\\path.npy', np.array([x_list, y_list, yaw_list]))
        ox, oy = self.ox, self.oy

        plt.plot(ox, oy, "sk", markersize=6 * 0.2)  # OBS_SIZE=0.2
        plt.plot(x_list, y_list, linewidth=1.5, color='r')

        plt.title("Hybrid A*")
        plt.axis("equal")

        plt.show()
        print("Plot one path!")