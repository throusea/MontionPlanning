import abc
import math
import sys
import time
from typing import Tuple, List

import numpy as np
import pathlib
from heapdict import heapdict
from irsim.world.object_base import ObstacleInfo
from matplotlib import pyplot as plt
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point

from planning_algorithm import PlanningAlgorithm, MapData

sys.path.append(str(pathlib.Path(__file__).parent.parent))
import CurvesGenerator.reeds_shepp as rs
import HybridAstarPlanner.astar as astar
import scipy.spatial.kdtree as kd


class HybridAstar(PlanningAlgorithm):


    def __init__(self, ackerman_model):
        super().__init__(ackerman_model)
        self.path = None
        self.ox = None
        self.oy = None


    class Node:
        def __init__(self, xind, yind, yawind, direction, x, y,
                     yaw, directions, steer, cost, pind):
            self.xind = xind
            self.yind = yind
            self.yawind = yawind
            self.direction = direction
            self.x = x
            self.y = y
            self.yaw = yaw
            self.directions = directions
            self.steer = steer
            self.cost = cost
            self.pind = pind

    class Para:
        def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                     xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
            self.minx = minx
            self.miny = miny
            self.minyaw = minyaw
            self.maxx = maxx
            self.maxy = maxy
            self.maxyaw = maxyaw
            self.xw = xw
            self.yw = yw
            self.yaww = yaww
            self.xyreso = xyreso
            self.yawreso = yawreso
            self.ox = ox
            self.oy = oy
            self.kdtree = kdtree

    class Path:
        def __init__(self, x, y, yaw, direction, cost):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.direction = direction
            self.cost = cost

    class QueuePrior:
        def __init__(self):
            self.queue = heapdict()

        def empty(self):
            return len(self.queue) == 0  # if Q is empty

        def put(self, item, priority):
            self.queue[item] = priority  # push

        def get(self):
            return self.queue.popitem()[0]  # pop out element with smallest priority


    @staticmethod
    def get_local_obstacles(obstacles: List[ObstacleInfo]):
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
        return ox, oy


    def run(self, map_data: MapData):
        start, goal, obstacle_list = map_data.start, map_data.goal, map_data.obstacles
        self.ox, self.oy = self.get_local_obstacles(obstacle_list)

        st = time.time()
        self.path = self.hybrid_astar_planning(
            start[0], start[1], start[2], goal[0], goal[1], goal[2],
            self.ox, self.oy,
            self.c.XY_RESO,
            self.c.YAW_RESO
        )
        et = time.time()
        self.metrics.planning_time = et - st
        if self.path:
            self.metrics.success_rate = 1
        else:
            self.metrics.success_rate = 0
        return self.path

    def hybrid_astar_planning(self, sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
        sxr, syr = round(sx / xyreso), round(sy / xyreso)
        gxr, gyr = round(gx / xyreso), round(gy / xyreso)
        syawr = round(rs.pi_2_pi(syaw) / yawreso)
        gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

        nstart = self.Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
        ngoal = self.Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

        kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
        P = self.calc_parameters(ox, oy, xyreso, yawreso, kdtree)

        hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
        steer_set, direc_set = self.calc_motion_set()
        open_set, closed_set = {self.calc_index(nstart, P): nstart}, {}

        qp = self.QueuePrior()
        qp.put(self.calc_index(nstart, P), self.calc_hybrid_cost(nstart, hmap, P))
        self.metrics.num_of_sampled_nodes = 0
        self.metrics.num_of_collision_check = 0
        self.metrics.num_of_sampled_nodes += 1

        while True:
            if not open_set:
                return None

            ind = qp.get()
            n_curr = open_set[ind]
            closed_set[ind] = n_curr
            open_set.pop(ind)

            update, fpath = self.update_node_with_analystic_expantion(n_curr, ngoal, P)

            if update:
                fnode = fpath
                break

            for i in range(len(steer_set)):
                node = self.calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

                if not node:
                    continue

                node_ind = self.calc_index(node, P)

                if node_ind in closed_set:
                    continue

                if node_ind not in open_set:
                    open_set[node_ind] = node
                    qp.put(node_ind, self.calc_hybrid_cost(node, hmap, P))
                    self.metrics.num_of_sampled_nodes += 1
                else:
                    if open_set[node_ind].cost > node.cost:
                        open_set[node_ind] = node
                        qp.put(node_ind, self.calc_hybrid_cost(node, hmap, P))
                    self.metrics.num_of_sampled_nodes += 1

        return self.extract_path(closed_set, fnode, nstart)

    def extract_path(self, closed, ngoal, nstart):
        rx, ry, ryaw, direc = [], [], [], []
        cost = 0.0
        node = ngoal

        while True:
            rx += node.x[::-1]
            ry += node.y[::-1]
            ryaw += node.yaw[::-1]
            direc += node.directions[::-1]
            cost += node.cost

            if self.is_same_grid(node, nstart):
                break

            node = closed[node.pind]

        rx = rx[::-1]
        ry = ry[::-1]
        ryaw = ryaw[::-1]
        direc = direc[::-1]

        direc[0] = direc[1]
        path = self.Path(rx, ry, ryaw, direc, cost)

        return path

    def calc_next_node(self, n_curr, c_id, u, d, P):
        step = self.c.XY_RESO * 2

        nlist = math.ceil(step / self.c.MOVE_STEP)
        xlist = [n_curr.x[-1] + d * self.c.MOVE_STEP * math.cos(n_curr.yaw[-1])]
        ylist = [n_curr.y[-1] + d * self.c.MOVE_STEP * math.sin(n_curr.yaw[-1])]
        yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * self.c.MOVE_STEP / self.c.WB * math.tan(u))]

        for i in range(nlist - 1):
            xlist.append(xlist[i] + d * self.c.MOVE_STEP * math.cos(yawlist[i]))
            ylist.append(ylist[i] + d * self.c.MOVE_STEP * math.sin(yawlist[i]))
            yawlist.append(rs.pi_2_pi(yawlist[i] + d * self.c.MOVE_STEP / self.c.WB * math.tan(u)))

        xind = round(xlist[-1] / P.xyreso)
        yind = round(ylist[-1] / P.xyreso)
        yawind = round(yawlist[-1] / P.yawreso)

        if not self.is_index_ok(xind, yind, xlist, ylist, yawlist, P):
            return None

        cost = 0.0

        if d > 0:
            direction = 1
            cost += abs(step)
        else:
            direction = -1
            cost += abs(step) * self.c.BACKWARD_COST

        if direction != n_curr.direction:  # switch back penalty
            cost += self.c.GEAR_COST

        cost += self.c.STEER_ANGLE_COST * abs(u)  # steer angle penalyty
        cost += self.c.STEER_CHANGE_COST * abs(n_curr.steer - u)  # steer change penalty
        cost = n_curr.cost + cost

        directions = [direction for _ in range(len(xlist))]

        node = self.Node(xind, yind, yawind, direction, xlist, ylist,
                    yawlist, directions, u, cost, c_id)

        return node

    def is_index_ok(self, xind, yind, xlist, ylist, yawlist, P):
        if xind <= P.minx or \
                xind >= P.maxx or \
                yind <= P.miny or \
                yind >= P.maxy:
            return False

        ind = range(0, len(xlist), self.c.COLLISION_CHECK_STEP)

        nodex = [xlist[k] for k in ind]
        nodey = [ylist[k] for k in ind]
        nodeyaw = [yawlist[k] for k in ind]

        if self.is_collision(nodex, nodey, nodeyaw, P):
            return False

        return True

    def update_node_with_analystic_expantion(self, n_curr, ngoal, P):
        path = self.analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

        if not path:
            return False, None

        fx = path.x[1:-1]
        fy = path.y[1:-1]
        fyaw = path.yaw[1:-1]
        fd = path.directions[1:-1]

        fcost = n_curr.cost + self.calc_rs_path_cost(path)
        fpind = self.calc_index(n_curr, P)
        fsteer = 0.0

        fpath = self.Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                     fx, fy, fyaw, fd, fsteer, fcost, fpind)

        return True, fpath

    def analystic_expantion(self, node, ngoal, P):
        sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
        gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

        maxc = math.tan(self.c.MAX_STEER) / self.c.WB
        paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=self.c.MOVE_STEP)
        # for path in paths:
        #     print(path)

        if not paths:
            return None

        pq = self.QueuePrior()
        for path in paths:
            pq.put(path, self.calc_rs_path_cost(path))

        while not pq.empty():
            path = pq.get()
            ind = range(0, len(path.x), self.c.COLLISION_CHECK_STEP)

            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]

            if not self.is_collision(pathx, pathy, pathyaw, P):
                return path

        return None

    def is_collision(self, x, y, yaw, P: Para):
        for ix, iy, iyaw in zip(x, y, yaw):
            self.metrics.num_of_collision_check += 1
            d = 0.2  # OBS_SIZE = 0.2
            dl = (self.c.RF - self.c.RB) / 2.0
            r = (self.c.RF + self.c.RB) / 2.0 + d + 0.1

            cx = ix + dl * math.cos(iyaw)
            cy = iy + dl * math.sin(iyaw)

            ids = P.kdtree.query_ball_point([cx, cy], r)

            if not ids:
                continue

            for i in ids:
                xo = P.ox[i] - cx
                yo = P.oy[i] - cy
                dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                if abs(dx) < r and abs(dy) < self.c.W / 2 + d:
                    return True

        return False

    def calc_rs_path_cost(self, rspath):
        cost = 0.0

        for lr in rspath.lengths:
            if lr >= 0:
                cost += 1
            else:
                cost += abs(lr) * self.c.BACKWARD_COST

        for i in range(len(rspath.lengths) - 1):
            if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
                cost += self.c.GEAR_COST

        for ctype in rspath.ctypes:
            if ctype != "S":
                cost += self.c.STEER_ANGLE_COST * abs(self.c.MAX_STEER)

        nctypes = len(rspath.ctypes)
        ulist = [0.0 for _ in range(nctypes)]

        for i in range(nctypes):
            if rspath.ctypes[i] == "R":
                ulist[i] = -self.c.MAX_STEER
            elif rspath.ctypes[i] == "WB":
                ulist[i] = self.c.MAX_STEER

        for i in range(nctypes - 1):
            cost += self.c.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

        return cost

    def calc_hybrid_cost(self, node, hmap, P):
        cost = node.cost + \
               self.c.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

        return cost

    def calc_motion_set(self):
        s = np.arange(self.c.MAX_STEER / self.c.N_STEER,
                      self.c.MAX_STEER, self.c.MAX_STEER / self.c.N_STEER)

        steer = list(s) + [0.0] + list(-s)
        direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
        steer = steer + steer

        return steer, direc

    def is_same_grid(self, node1, node2):
        if node1.xind != node2.xind or \
                node1.yind != node2.yind or \
                node1.yawind != node2.yawind:
            return False

        return True

    def calc_index(self, node, P):
        ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
              (node.yind - P.miny) * P.xw + \
              (node.xind - P.minx)

        return ind

    def calc_parameters(self, ox, oy, xyreso, yawreso, kdtree):
        minx = round(min(ox) / xyreso)
        miny = round(min(oy) / xyreso)
        maxx = round(max(ox) / xyreso)
        maxy = round(max(oy) / xyreso)

        xw, yw = maxx - minx, maxy - miny

        minyaw = round(-self.c.PI / yawreso) - 1
        maxyaw = round(self.c.PI / yawreso)
        yaww = maxyaw - minyaw

        return self.Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                    xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)

    def plot_path(self, save_path=None):
        # np.save('..\\..\\path.npy', np.array([x_list, y_list, yaw_list]))
        ox, oy = self.ox, self.oy

        plt.plot(ox, oy, "sk", markersize=6 * 0.2)  # OBS_SIZE=0.2

        if self.path is not None:
            x_list = self.path.x
            y_list = self.path.y
            yaw_list = self.path.yaw
            direction = self.path.direction

            # compute the path length
            las_x, las_y = None, None
            path_len = 0
            for x, y in zip(x_list, y_list):
                if las_x and las_y:
                    path_len += math.sqrt((x - las_x) ** 2 + (y - las_y) ** 2)
                las_x, las_y = x, y
            self.metrics.path_length = path_len

            # plot the path
            plt.plot(x_list, y_list, linewidth=1.5, color='r')

        plt.title("Hybrid A*")
        plt.axis("equal")

        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path)
        print("Plot one path!")
