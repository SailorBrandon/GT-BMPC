import math
import motion_primitive_planner.parameters as parameters


dt = 0.1

class Trajectory:
    def __init__(self) -> None:
        self.t = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        
        self.y = []
        self.y_d = []
        self.y_dd = []
        self.x = []
        self.x_d = []
        self.x_dd = []
        
        self.heading = []
        self.vel = []
        self.acc_long = []
        self.acc_lat = []
        self.steer = []
        self.curvature = []
        self.jerk = []
        self.angular_acc = []
        
        self.self_oriented_cost = 0
        self.mutual_cost = 0
        self.cost_jerk = 0
        self.cost_target_lane = 0
        self.cost_efficiency = 0
        self.y_terminal = None
        self.v_terminal = None
        
    def get_state_from_flat_output(self, vehicle_wheel_base):
        self.vel = [math.hypot(x_d, y_d) for x_d, y_d in zip(self.x_d, self.y_d)]
        self.acc_long = [(x_d*x_dd + y_d*y_dd) / math.hypot(x_d, y_d) for x_d, y_d, x_dd, y_dd in zip(self.x_d, self.y_d, self.x_dd, self.y_dd)]
        self.acc_lat = [(x_d*y_dd - y_d*x_dd) / math.hypot(x_d, y_d) for x_d, y_d, x_dd, y_dd in zip(self.x_d, self.y_d, self.x_dd, self.y_dd)]
        self.heading = [math.atan2(y_d, x_d) for x_d, y_d in zip(self.x_d, self.y_d)]
        self.steer = [math.atan((x_d*y_dd - y_d*x_dd) * vehicle_wheel_base / (math.hypot(x_d, y_d))**3) for x_d, y_d, x_dd, y_dd in zip(self.x_d, self.y_d, self.x_dd, self.y_dd)]
        self.curvature = [(x_d*y_dd - y_d*x_dd) / (math.hypot(x_d, y_d))**3 for x_d, y_d, x_dd, y_dd in zip(self.x_d, self.y_d, self.x_dd, self.y_dd)]
        
import numpy as np
from math import cos, sin

def find_vehicle_vertices(rear_wheel_center, heading, length, width, vehicle_wheel_base):        
    center = rear_wheel_center + vehicle_wheel_base / 2 * np.array([cos(heading), sin(heading)])
    inflation = parameters.collision_check_inflation
    half_veh_len = length / 2 * inflation
    half_veh_wid = width / 2 * inflation
    
    R = np.array([[cos(heading), -sin(heading)],
                    [sin(heading), cos(heading)]])
    return [center + R @ np.array([-half_veh_len, -half_veh_wid]),
            center + R @ np.array([+half_veh_len, -half_veh_wid]),
            center + R @ np.array([+half_veh_len, +half_veh_wid]),
            center + R @ np.array([-half_veh_len, +half_veh_wid])]

def is_separating_axis(n, P1, P2):
    min1, max1 = float('+inf'), float('-inf')
    min2, max2 = float('+inf'), float('-inf')
    for v in P1:
        proj = np.dot(v, n)
        min1 = min(min1, proj)
        max1 = max(max1, proj)
    for v in P2:
        proj = np.dot(v, n)
        min2 = min(min2, proj)
        max2 = max(max2, proj)
    
    if max1 >= min2 and max2 >= min1:
        return False
    
    return True

def find_edges_norms(P):
    edges = []
    norms = []
    num_edge = len(P)
    for i in range(num_edge):
        edge = P[(i + 1)%num_edge] - P[i]
        norm = np.array([-edge[1], edge[0]])
        edges.append(edge)
        norms.append(norm)
    return edges, norms
        
def collide(P1, P2):
    """
    Check if two polygons overlap.
    We follow https://hackmd.io/@US4ofdv7Sq2GRdxti381_A/ryFmIZrsl?type=view
    Args:
        p1 (List): List of the vertices of a polygon
        p2 (List): List of the vertices of a polygon
    """
    
    P1 = [np.array(v, 'float64') for v in P1]
    P2 = [np.array(v, 'float64') for v in P2]
    
    _, norms1 = find_edges_norms(P1)
    _, norms2 = find_edges_norms(P2)
    norms = norms1 + norms2
    for n in norms:
        if is_separating_axis(n, P1, P2):
            return False
        
    return True


# three ways of collision checking
# 1. 名可夫斯基距离
# 2....collision avoidance paper
# 3. 分离轴定理