"""

Path planning Sample Code with RRT with Reeds-Shepp path

author: AtsushiSakai(@Atsushi_twi)

"""
import copy
import math
import os
import random
import sys
import pathlib
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from irsim.lib.kinematics import ackermann_kinematics
from irsim.world.object_base import ObstacleInfo
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from RRTStarReedsShepp.my_rrt_reeds_shepp import MyRRTStar
from RRT.rrt import RRT
from RRT.rrt_with_pathsmoothing import path_smoothing
from RRTStar.rrt_star import RRTStar
from BSplinePath.bspline_path import approximate_b_spline_path
from CubicSpline.cubic_spline_planner import CubicSpline2D
from planning_algorithm import PlanningAlgorithm, MapData, TestMetrics

show_animation = True


class MyRRTStarReedsShepp(PlanningAlgorithm):

    def __init__(self, ackerman_model):
        super().__init__(ackerman_model)
        self.rrt = None
        self.path = None
        self.obstacle_list = None

    def run(self, map_data: MapData):
        start, goal, obstacle_list = map_data.start, map_data.goal, map_data.obstacles
        self.obstacle_list = self.get_local_obstacles(obstacle_list)
        # self.rrt = MyRRTStar(start, goal, obstacle_list, [0, 100], max_iter=50000, ackerman_model=self.c)
        self.rrt = RRTStar(start, goal, self.obstacle_list, [0, 100], max_iter=10000, expand_dis=2.0, robot_radius=2.0)

        st = time.time()
        self.path = self.rrt.planning(animation=False)
        et = time.time()
        # self.rrt = RRTStarReedsShepp(start, goal, obstacle_list, [])
        self.metrics.num_of_sampled_nodes = len(self.rrt.node_list)
        self.metrics.planning_time = et - st
        self.metrics.num_of_collision_check = self.rrt.num_of_collision_check
        if self.path:
            self.metrics.success_rate = 1
        else:
            self.metrics.success_rate = 0
        return self.path

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
        # self.ox, self.oy = ox, oy
        new_obstacles = zip(ox, oy, [0.2] * len(ox))
        return list(new_obstacles)


    def plot_path(self, save_path=None):

        if self.path:
            print("Searching succeed!")
            # Path smoothing
            maxIter = 1000
            smoothedPath = path_smoothing(self.path, maxIter, self.obstacle_list)

            x_list = [p[0] for p in smoothedPath]
            y_list = [p[1] for p in smoothedPath]

            rax_list, ray_list = x_list, y_list
            # plt.plot(rax_list, ray_list, linewidth=1.5, color='b', label='Path0')

            ds = 0.1  # [m] distance of each interpolated points
            # print(len(rax_list))
            # sp = CubicSpline2D(rax_list, ray_list)
            # s = np.arange(0, sp.s[-1], ds)
            # rx, ry, ryaw, rk = [], [], [], []
            # for i_s in s:
            #     ix, iy = sp.calc_position(i_s)
            #     rx.append(ix)
            #     ry.append(iy)
            #     ryaw.append(sp.calc_yaw(i_s))
            #     rk.append(sp.calc_curvature(i_s))
            # rax_list, ray_list = rx, ry
            # rax_list, ray_list, heading_list, curvature = approximate_b_spline_path(
            #     x_list, y_list, len(x_list)*8, s=0.5
            # )

            # compute the path length
            las_x, las_y = None, None
            path_len = 0
            for x, y in zip(rax_list, ray_list):
                if las_x and las_y:
                    path_len += math.sqrt((x - las_x) ** 2 + (y - las_y) ** 2)
                las_x, las_y = x, y
            self.metrics.path_length = path_len
            # plot the path
            plt.plot(rax_list, ray_list, linewidth=1.5, color='r', label='Path')

        else:
            print("Searching failed!")

        plt.title("RRT*")
        plt.axis("equal")

        if not save_path:
            plt.show()
        else:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            print(f"Plot one path to {save_path}!")