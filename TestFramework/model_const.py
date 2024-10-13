import math

import numpy as np


class C:  # Parameter config
    PI = math.pi

    XY_RESO = 2.0  # [m]
    YAW_RESO = np.deg2rad(7.5)  # [rad]
    MOVE_STEP = 0.1  # [m] path interporate resolution
    N_STEER = 20.0  # steer command number
    COLLISION_CHECK_STEP = 5  # skip number for collision check
    EXTEND_BOUND = 1  # collision check range extended

    GEAR_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    H_COST = 15.0  # Heuristic cost penalty cost

    RF = 3.7  #  4.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.9  # 1.0  # [m] distance from rear to vehicle back end of vehicle
    W = 1.6  # 3.0  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 3   # 3.5  # [m] Wheel base
    TR = 0.5  # [m] Tyre radius
    TW = 1  # [m] Tyre width
    MAX_STEER = 0.6  # [rad] maximum steering angle
