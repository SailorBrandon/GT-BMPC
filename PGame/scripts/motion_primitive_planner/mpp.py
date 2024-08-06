import numpy as np
import math
from itertools import combinations
import time
import rospy
from scipy.spatial import KDTree
from motion_primitive_planner.math.cubic_spline import Spline2D
from motion_primitive_planner.math.polynomial import QuarticPolynomial, QuinticPolynomial
from motion_primitive_planner.utils import Trajectory, collide, find_vehicle_vertices
import motion_primitive_planner.parameters as parameters
from copy import deepcopy

import pickle

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class MPP:
    def __init__(self, lane_dict, vehicle_states):
        self._lane_dict = lane_dict
        self._set_reference_line()
        self._vehicle_states_cartesian = vehicle_states
        self._planner_name = None
        # self._all_ego_decisions = None
        # self._target_vehicle = None
        # self._vehicle_wheel_bases = None
        # self._vehicle_lengths = None
        # self._vehicle_widths = None
        # self._desired_velocities = None
        # self._last_vehicle_states_cartesian = None
        # self._vehicle_states_frenet = {}
        
    def reset(self):
        self._all_ego_decisions = None
        self._target_vehicle = None
        self._vehicle_wheel_bases = None
        self._vehicle_lengths = None
        self._vehicle_widths = None
        self._desired_velocities = None
        self._last_vehicle_states_cartesian = None
        self._vehicle_states_frenet = {}
        
    def run_step(self, vehicle_wheel_bases, desired_velocities, vehicle_lengths, vehicle_widths):
        self._vehicle_lengths = vehicle_lengths
        self._vehicle_widths = vehicle_widths
        self._desired_velocities = desired_velocities
        self._vehicle_wheel_bases = vehicle_wheel_bases
        start_time = time.time()
        for name in self._vehicle_states_cartesian:
            if self._last_vehicle_states_cartesian is not None:
                self._vehicle_states_frenet[name] = self._get_frenet_state(self._vehicle_states_cartesian[name], self._last_vehicle_states_cartesian[name])
            else:
                self._vehicle_states_frenet[name] = self._get_frenet_state(self._vehicle_states_cartesian[name], None)
        rospy.loginfo("vehicle_states_cartesian %s", self._vehicle_states_cartesian)
        rospy.loginfo("vehicle_states_frenet %s", self._vehicle_states_frenet)
        all_trajectories = self._get_trajectory_by_motion_primitive()
        rospy.loginfo("Time for getting trajectory: %.2f s", time.time() - start_time)
        start_time = time.time()
        best_trajectory = self._trajectory_selection(all_trajectories)
        if best_trajectory is None:
            rospy.logwarn("No valid trajectory found, brake forcefully!")
            acc = -2
            steer = 0
            return acc, steer, None
        rospy.loginfo("Time for trajectory selection: %.2f s", time.time() - start_time)
        rospy.loginfo("(unweighted) cost_jerk: %.4f, cost_target_lane: %.4f, cost_efficiency: %.4f", best_trajectory.cost_jerk, best_trajectory.cost_target_lane, best_trajectory.cost_efficiency)
        acc = best_trajectory.acc_long[1]
        steer = best_trajectory.steer[1]
        # acc = best_trajectory.acc_long[5]
        # steer = best_trajectory.steer[5]
        # acc = best_trajectory.acc_long[0]
        # steer = best_trajectory.steer[0]
        self._last_vehicle_states_cartesian = deepcopy(self._vehicle_states_cartesian)
        
        return acc, steer, best_trajectory

    def _set_reference_line(self):
        self.csp = Spline2D(self._lane_dict["target_guideline"][:, 0],
                            self._lane_dict["target_guideline"][:, 1])
        self.s = np.arange(0, self.csp.s[-1], 0.1)
        self.rx, self.ry, self.ryaw, self.rk = [], [], [], []
        for i_s in self.s:
            ix, iy = self.csp.calc_position(i_s)
            self.rx.append(ix)
            self.ry.append(iy)
            self.ryaw.append(self.csp.calc_yaw(i_s))
            self.rk.append(self.csp.calc_curvature(i_s))
        point_xy = np.array([item for item in zip(self.rx, self.ry)])
        self.ref_kdtree = KDTree(point_xy)

    def _get_frenet_state(self, state_cartesian, state_cartesian_prev):
        # _, idx_r = self.ref_kdtree.query([state_cartesian[0], state_cartesian[1]])
        # if idx_r >= len(self.s): # TODO
        #     idx_r = len(self.s) - 1
        # s_r = self.s[idx_r]
        # x_r, y_r = self.csp.calc_position(s_r)
        # k_r = self.csp.calc_curvature(s_r)
        # yaw_r = self.csp.calc_yaw(s_r)
        # dyaw_r = self.csp.calc_curvature_d(s_r)
        # delta_theta = state_cartesian[2] - yaw_r
        # # k_x: curvature of vehicle's route
        # if state_cartesian is not None and state_cartesian_prev is not None:
        #     dx = state_cartesian[0] - state_cartesian_prev[0]
        #     dy = state_cartesian[1] - state_cartesian_prev[1]
        #     dyaw = state_cartesian[2] - state_cartesian_prev[2]
        #     ds = math.hypot(dx, dy)
        #     if 0 < ds:
        #         k_x = dyaw / math.hypot(dx, dy)
        #     else:
        #         k_x = None
        # else:
        #     k_x = None
        # s = s_r
        # x_delta = state_cartesian[0] - x_r
        # y_delta = state_cartesian[1] - y_r
        # d = np.sign(y_delta * math.cos(yaw_r) - x_delta * math.sin(yaw_r)) * math.hypot(x_delta, y_delta)
        # d_d = state_cartesian[3] * math.sin(delta_theta)
        # coeff_1 = 1 - k_r * d
        # acc = 0
        # if state_cartesian_prev is not None:
        #     acc = (state_cartesian[3] - state_cartesian_prev[3]) / parameters.dt
        # d_dd = acc * math.sin(delta_theta)
        # s_d = state_cartesian[3] * math.cos(delta_theta) / coeff_1
                
        # if k_x is None:
        #     s_dd = 0
        # else:
        #     # s_ds = coeff_1 * math.tan(delta_theta)
        #     # coeff_2 = coeff_1 / math.cos(delta_theta) * k_x - k_r
        #     # coeff_3 = dyaw_r * d + yaw_r * s_ds
        #     # s_dd = state_cartesian[4] * math.cos(delta_theta) - \
        #     #     (s_d ** 2) * (s_ds * coeff_2 - coeff_3) / coeff_1
           
        #     kappa_r = self.csp.calc_curvature(s_r)
        #     s_ds = coeff_1 * math.tan(delta_theta)
        #     coeff_2 = coeff_1 / math.cos(delta_theta) * k_x - k_r
        #     coeff_3 = dyaw_r * d + kappa_r * s_ds
        #     s_dd = (acc * math.cos(delta_theta) -
        #         (s_d ** 2) * (s_ds * coeff_2 - coeff_3)) / coeff_1
                    
        ## Referring to 
        # https://blog.csdn.net/u013468614/article/details/108748016
        _, idx_r = self.ref_kdtree.query([state_cartesian[0], state_cartesian[1]])
        if idx_r >= len(self.s): # TODO
            idx_r = len(self.s) - 1
        x_x = state_cartesian[0]
        y_x = state_cartesian[1]
        theta_x = state_cartesian[2]
        v_x = state_cartesian[3]
        # a_x = state_cartesian[4]
        a_x = 0
        if state_cartesian_prev is not None:
            a_x = (v_x - state_cartesian_prev[3]) / parameters.dt
        k_x = 0
        if state_cartesian_prev is not None:
            distance = math.hypot(x_x - state_cartesian_prev[0], y_x - state_cartesian_prev[1])
            dtheta = theta_x - state_cartesian_prev[2]
            if distance != 0:
                k_x = dtheta / distance
        
        s_r = self.s[idx_r]
        x_r, y_r = self.csp.calc_position(s_r)
        theta_r = self.csp.calc_yaw(s_r)
        k_r = self.csp.calc_curvature(s_r)
        k_r_d = self.csp.calc_curvature_d(s_r)

        delta_x = x_x - x_r
        delta_y = y_x - y_r        
        delta_theta = theta_x - theta_r

        C_drdx = delta_y * math.cos(theta_r) - delta_x * math.sin(theta_r)
        
        d = np.sign(C_drdx) * math.hypot(delta_x, delta_y)
        C_4 = 1 - k_r * d
        d_p = C_4 * math.tan(delta_theta)
        C_6 = k_r_d * d + k_r * d_p
        k_x = 0
        if state_cartesian_prev is not None:
            distance = math.hypot(x_x - state_cartesian_prev[0], y_x - state_cartesian_prev[1])
            dtheta = pi_2_pi(theta_x - state_cartesian_prev[2])
            if distance != 0:
                # print(f"distance: {distance}")
                # print(f"dtheta: {dtheta}")
                # print(f"theta_x: {theta_x}")
                # print(f"state_cartesian_prev[2]: {state_cartesian_prev[2]}")
                k_x = dtheta / distance
        d_pp = -C_6 * math.tan(delta_theta) + C_4 / math.cos(delta_theta)**2 * (k_x * C_4 / math.cos(delta_theta) - k_r)
        
        s = s_r
        s_d = v_x * math.cos(delta_theta) / C_4
        C_10 = k_x * C_4 / math.cos(delta_theta) - k_r
        s_dd = (a_x * math.cos(delta_theta) - (s_d**2) * (d_p * C_10 - C_6)) / C_4
        
        d_d = s_d * d_p
        d_dd = s_dd * d_p + s_d**2 * d_pp
        
        # print(f"a_x: {a_x}")
        # print(f"delta_theta: {delta_theta}")
        # print(f"d_p: {d_p}")
        # print(f"C_4: {C_4}")
        # print(f"C_6: {C_6}")
        # print(f"C_10: {C_10}")
        # print(f"k_x: {k_x}")
        # print(f"k_r: {k_r}")
        # print(f"s_d**2: {s_d**2}")
        # print(f"s: {s}, s_d: {s_d}, s_dd: {s_dd}, d: {d}, d_d: {d_d}, d_dd: {d_dd}\n\n")
        
        return s, s_d, s_dd, d, d_d, d_dd

    # def _get_cartesian_trajectories(self, trajectory_list, vehicle_wheel_base):
    #     valid_trajectory_list = []
    #     for trajectory in trajectory_list:   
    #         for s, s_d, s_dd, d, d_d, d_dd in zip(trajectory.s, trajectory.s_d, trajectory.s_dd, trajectory.d, trajectory.d_d, trajectory.d_dd):
    #             ix, iy = self.csp.calc_position(s)
    #             if ix is None or iy is None:
    #                 rospy.logwarn("ix or iy is None with s: %s", s)
    #                 break
    #             i_yaw = self.csp.calc_yaw(s)
    #             trajectory.x.append(ix - d * math.sin(i_yaw))                 
    #             trajectory.y.append(iy + d * math.cos(i_yaw))
    #             trajectory.x_d.append(s_d * math.cos(i_yaw) - d_d * math.sin(i_yaw))
    #             trajectory.y_d.append(s_d * math.sin(i_yaw) + d_d * math.cos(i_yaw))
    #             trajectory.x_dd.append(s_dd * math.cos(i_yaw) - d_dd * math.sin(i_yaw))
    #             trajectory.y_dd.append(s_dd * math.sin(i_yaw) + d_dd * math.cos(i_yaw))
    #         trajectory.get_state_from_flat_output(vehicle_wheel_base)
    #         valid_trajectory_list.append(trajectory)
    #     return valid_trajectory_list

    def _get_cartesian_trajectories(self, trajectory_list, vehicle_wheel_base):
        valid_trajectory_list = []
        for trajectory in trajectory_list:   
            is_valid = True
            for s, s_d, s_dd, d, d_d, d_dd in zip(trajectory.s, trajectory.s_d, trajectory.s_dd, trajectory.d, trajectory.d_d, trajectory.d_dd):
                ix, iy = self.csp.calc_position(s)
                if ix is None or iy is None:
                    rospy.logwarn("ix or iy is None with s: %s", s)
                    break
                if s_d == 0:  raise ValueError("s_d is 0")
                d_p = d_d / s_d
                d_pp = (d_dd - s_dd * d_p) / s_d**2
                
                i_yaw = self.csp.calc_yaw(s)
                trajectory.x.append(ix - d * math.sin(i_yaw))                 
                trajectory.y.append(iy + d * math.cos(i_yaw))

                i_kappa = self.csp.calc_curvature(s)
                vel = math.sqrt((1 - d * i_kappa)**2 * s_d**2 + d_d**2)
                trajectory.vel.append(vel)
    
                delta_theta = math.asin(d_d / vel)
                heading = i_yaw + delta_theta
                trajectory.heading.append(heading)          
                
                i_kappa_d = self.csp.calc_curvature_d(s)
                M1 = 1 - d * i_kappa
                M2 = math.cos(delta_theta)                
                M3 = M1 / M2
                M4 = M1 / M2**2
                curvature = ((d_pp + (i_kappa_d * d + i_kappa * d_p) * math.tan(delta_theta)) / M4 + i_kappa) / M3
                trajectory.curvature.append(curvature)
                steer = math.atan(vehicle_wheel_base * curvature)
                if steer > math.pi / 6:
                    is_valid = False
                    break
                trajectory.steer.append(steer)
                
                M5 = (1 - d * i_kappa) * math.tan(delta_theta) 
                M6 = curvature * M3 - i_kappa 
                acc = s_dd * M3 + s_d**2 / M2 * (M5 * M6 - (i_kappa_d * d + i_kappa * d_p))
                trajectory.acc_long.append(acc)
            if is_valid:
                valid_trajectory_list.append(trajectory)
        return valid_trajectory_list

    def _get_trajectory_by_motion_primitive(self):
        all_trajectories = {}
        for name, state_frenet in self._vehicle_states_frenet.items():
            if name == 0:
                all_trajectories[name] = self._get_cartesian_trajectories(self._get_motion_primitives(state_frenet, True),
                                                                          vehicle_wheel_base=self._vehicle_wheel_bases[name])
            else:
                all_trajectories[name] = self._get_cartesian_trajectories(self._get_motion_primitives(state_frenet, False),
                                                                          vehicle_wheel_base=self._vehicle_wheel_bases[name])
        return all_trajectories
        
    def _get_motion_primitives(self, state_frenet, sampling_lateral: bool):
        s, s_d, s_dd, d, d_d, d_dd = state_frenet
        trajectory_list = []
        def get_trajectory_properties(long_qp, trajectory):
            trajectory.s = [long_qp.calc_point(t) for t in trajectory.t]
            trajectory.s_d = [long_qp.calc_first_derivative(t) for t in trajectory.t]
            trajectory.s_dd = [long_qp.calc_second_derivative(t) for t in trajectory.t]
            trajectory.s_ddd = [long_qp.calc_third_derivative(t) for t in trajectory.t]
        for v_terminal in np.arange(parameters.MIN_SPEED, parameters.MAX_SPEED + 0.5*parameters.D_V, parameters.D_V):
            # for Ti in np.arange(parameters.MIN_T, parameters.MAX_T + 0.5 * parameters.DT, parameters.DT):
                long_qp = QuarticPolynomial(s, s_d, s_dd, v_terminal, 0, parameters.T)
                # long_qp = QuarticPolynomial(s, s_d, s_dd, v_terminal, 0, Ti)
                time_points = np.arange(0, parameters.T, parameters.dt).tolist()
                # time_points = np.arange(0, Ti, parameters.dt).tolist()
                if sampling_lateral:
                    for y_terminal in np.arange(parameters.MIN_LAT_DIST, parameters.MAX_LAT_DIST + 0.5*parameters.D_LAT_DIST, parameters.D_LAT_DIST):
                        lat_qp = QuinticPolynomial(d, d_d, d_dd, y_terminal, 0, 0, parameters.T)
                        # lat_qp = QuinticPolynomial(d, d_d, d_dd, y_terminal, 0, 0, Ti)
                        trajectory = Trajectory()
                        trajectory.y_terminal = y_terminal
                        trajectory.t = time_points
                        trajectory.d = [lat_qp.calc_point(t) for t in trajectory.t]
                        trajectory.d_d = [lat_qp.calc_first_derivative(t) for t in trajectory.t]
                        trajectory.d_dd = [lat_qp.calc_second_derivative(t) for t in trajectory.t]
                        trajectory.d_ddd = [lat_qp.calc_third_derivative(t) for t in trajectory.t]
                        trajectory.v_terminal = v_terminal
                        get_trajectory_properties(long_qp, trajectory)
                        trajectory_list.append(trajectory)
                        trajectory.cost_target_lane = sum(abs(np.array(trajectory.d))) / len(time_points) # assume the reference line is the target lane
                else:
                    trajectory = Trajectory()
                    trajectory.t = time_points
                    trajectory.v_terminal = v_terminal
                    trajectory.d = [d for _ in trajectory.t]
                    trajectory.d_d = [0 for _ in trajectory.t]
                    trajectory.d_dd = [0 for _ in trajectory.t]
                    trajectory.d_ddd = [0 for _ in trajectory.t]
                    get_trajectory_properties(long_qp, trajectory)
                    trajectory_list.append(trajectory)
        return trajectory_list        

    def _trajectory_selection(self, all_trajectories):
        if len(all_trajectories[0]) == 0:
            return None
        time_start = time.time()
        cost_matrix, potential_function_matrix = self._get_trajectory_cost(all_trajectories)
        # print("Time for calculating cost matrix: %.2f s" % (time.time() - time_start))
        # non_ego_decision_id, ego_decision_id = self._get_ne_potential_game(cost_matrix, potential_function_matrix)
        all_decision_ids = self._get_ne_matrix_game(cost_matrix)
        time_start = time.time()
        # print("Time for getting Nash Equilibrium: %.2f s" % (time.time() - time_start))
        best_ego_decision_id = all_decision_ids[0]
        # rospy.loginfo("vehicle 0 decision lateral: %s, longitudinal: %s", all_trajectories[0][best_ego_decision_id].y_terminal, all_trajectories[0][best_ego_decision_id].v_terminal)
        for i in [0, 1, 2]:
            rospy.loginfo("vehicle %s decision lateral: %s, longitudinal: %s", i, all_trajectories[i][all_decision_ids[i]].y_terminal, all_trajectories[i][all_decision_ids[i]].v_terminal)
        rospy.loginfo('potential function value: %s', potential_function_matrix[all_decision_ids])
        # best_ego_decision_id = self._get_naive_solution(all_trajectories[0])
        if potential_function_matrix[all_decision_ids] > 1000:  # TODO: tmp hard code
            return None
        return all_trajectories[0][best_ego_decision_id]
    
    def _get_trajectory_cost(self, all_trajectories):
        assert all_trajectories.keys() == {0, 1, 2}
        num_players = 3
        cost_matrix = np.zeros((len(all_trajectories[0]), len(all_trajectories[1]), len(all_trajectories[2]), num_players))
        potential_function_matrix = np.zeros((len(all_trajectories[0]), len(all_trajectories[1]), len(all_trajectories[2]), 1))
        
        for name, trajectories in all_trajectories.items():
            for trajectory in trajectories:
                trajectory.self_oriented_cost = self._get_self_oriented_cost(trajectory, name)
                
        mutual_cost_matrix_01 = np.zeros((len(all_trajectories[0]), len(all_trajectories[1])))
        mutual_cost_matrix_02 = np.zeros((len(all_trajectories[0]), len(all_trajectories[2])))
        mutual_cost_matrix_12 = np.zeros((len(all_trajectories[1]), len(all_trajectories[2])))
        for i in range(len(all_trajectories[0])):
            for j in range(len(all_trajectories[1])):
                pair_action_map = {0: i, 1: j}
                mutual_cost_matrix_01[i][j] = self._get_mutual_cost(all_trajectories, pair_action_map)
        for i in range(len(all_trajectories[0])):
            for k in range(len(all_trajectories[2])):
                pair_action_map = {0: i, 2: k}
                mutual_cost_matrix_02[i][k] = self._get_mutual_cost(all_trajectories, pair_action_map)
        for j in range(len(all_trajectories[1])):
            for k in range(len(all_trajectories[2])):
                pair_action_map = {1: j, 2: k}
                mutual_cost_matrix_12[j][k] = self._get_mutual_cost(all_trajectories, pair_action_map)
        
        alpha = 1
        beta = 1
        for i in range(len(all_trajectories[0])):
            for j in range(len(all_trajectories[1])):
                for k in range(len(all_trajectories[2])):
                    cost_matrix[i][j][k][0] = alpha * all_trajectories[0][i].self_oriented_cost + beta * (mutual_cost_matrix_01[i][j] + mutual_cost_matrix_02[i][k])
                    cost_matrix[i][j][k][1] = alpha * all_trajectories[1][j].self_oriented_cost + beta * (mutual_cost_matrix_01[i][j] + mutual_cost_matrix_12[j][k])
                    cost_matrix[i][j][k][2] = alpha * all_trajectories[2][k].self_oriented_cost + beta * (mutual_cost_matrix_02[i][k] + mutual_cost_matrix_12[j][k])
                    potential_function_matrix[i][j][k] = alpha * (all_trajectories[0][i].self_oriented_cost + all_trajectories[1][j].self_oriented_cost + all_trajectories[2][k].self_oriented_cost) + \
                                                         beta * (mutual_cost_matrix_01[i][j] + mutual_cost_matrix_02[i][k] + mutual_cost_matrix_12[j][k])
        # pickle.dump(all_trajectories, open("/home/kasm-user/workspace/tcst_ws/all_trajectories.pkl", "wb"))
        # pickle.dump(self._vehicle_lengths, open("/home/kasm-user/workspace/tcst_ws/vehicle_lengths.pkl", "wb"))
        # pickle.dump(self._vehicle_widths, open("/home/kasm-user/workspace/tcst_ws/vehicle_widths.pkl", "wb"))
        # pickle.dump(self._vehicle_wheel_bases, open("/home/kasm-user/workspace/tcst_ws/vehicle_wheel_bases.pkl", "wb"))
        # np.save("/home/kasm-user/workspace/tcst_ws/cost_matrix.npy", cost_matrix)
        # np.save("/home/kasm-user/workspace/tcst_ws/potential_function_matrix.npy", potential_function_matrix)
        return cost_matrix, potential_function_matrix
    
    def _get_self_oriented_cost(self, trajectory, name):
        cost_weights = parameters.all_cost_weights[name]
        num_sim_step = len(trajectory.x)
        cost_jerk = sum([((acc - acc_prev)/parameters.dt)**2 for acc, acc_prev in zip(trajectory.acc_long, trajectory.acc_long[1:])]) / num_sim_step
        target_speed = self._desired_velocities[name]        
        cost_efficiency = sum([(vel - target_speed)**2 for vel in trajectory.vel]) / num_sim_step
        trajectory.jerk = [abs((acc - acc_prev)/parameters.dt) for acc, acc_prev in zip(trajectory.acc_long, trajectory.acc_long[1:])]
        trajectory.angular_acc = [abs((heading - heading_prev)/parameters.dt) for heading, heading_prev in zip(trajectory.heading, trajectory.heading[1:])]
        trajectory.cost_jerk = cost_jerk
        trajectory.cost_efficiency = cost_efficiency
        return trajectory.cost_jerk * cost_weights["jerk"] + trajectory.cost_target_lane * cost_weights["target_lane"] \
            + trajectory.cost_efficiency * cost_weights["efficiency"]
    
    def _get_mutual_cost(self, all_trajectories, pair_action_map):
        name_0, name_1 = pair_action_map.keys()
        num_sim_step = min(len(all_trajectories[name_0][pair_action_map[name_0]].x), len(all_trajectories[name_1][pair_action_map[name_1]].x))
        mutual_cost = 0
        prox_cost = 0
        dist_prox = 1
        for i in range(num_sim_step):
            pos_0 = np.array([all_trajectories[name_0][pair_action_map[name_0]].x[i], all_trajectories[name_0][pair_action_map[name_0]].y[i]])
            pos_1 = np.array([all_trajectories[name_1][pair_action_map[name_1]].x[i], all_trajectories[name_1][pair_action_map[name_1]].y[i]])
            prox_cost += 1 / (np.linalg.norm(pos_0 - pos_1) + 1e-4)
            # dist = np.linalg.norm(pos_0 - pos_1)
                
        prox_cost /= num_sim_step   
        
        collision_cost = 0
        for i in range(num_sim_step):
            vertices_0 = find_vehicle_vertices((all_trajectories[name_0][pair_action_map[name_0]].x[i], all_trajectories[name_0][pair_action_map[name_0]].y[i]),
                                                all_trajectories[name_0][pair_action_map[name_0]].heading[i],
                                                self._vehicle_lengths[name_0],
                                                self._vehicle_widths[name_0],
                                                self._vehicle_wheel_bases[name_0])
            vertices_1 = find_vehicle_vertices((all_trajectories[name_1][pair_action_map[name_1]].x[i], all_trajectories[name_1][pair_action_map[name_1]].y[i]),
                                                all_trajectories[name_1][pair_action_map[name_1]].heading[i],
                                                self._vehicle_lengths[name_1],
                                                self._vehicle_widths[name_1],
                                                self._vehicle_wheel_bases[name_1])
            if collide(vertices_0, vertices_1):
                collision_cost = 1000
                break
    
        mutual_cost = prox_cost * parameters.all_cost_weights["proximity"] + collision_cost
        return mutual_cost    
    
    def _get_naive_solution(self, ego_trajectories: list) -> int:
        best_idx = None
        best_cost = float('inf')
        for idx, traj in enumerate(ego_trajectories):
            if traj.self_oriented_cost < best_cost:
                best_idx = idx
                best_cost = traj.self_oriented_cost
        return best_idx
            
    def _get_ne_matrix_game(self, cost_matrix: np.ndarray) -> tuple:    
        NEs = []
        num_rows, num_cols, num_depth = cost_matrix.shape[:3]
        for i in range(num_rows):
            for j in range(num_cols):
                for k in range(num_depth):
                    player1_cost = cost_matrix[i, j, k][0]
                    player2_cost = cost_matrix[i, j, k][1]
                    player3_cost = cost_matrix[i, j, k][2]
                    is_best_response_p1 = all(player1_cost <= cost_matrix[x, j, k][0] for x in range(num_rows))
                    is_best_response_p2 = all(player2_cost <= cost_matrix[i, y, k][1] for y in range(num_cols))
                    is_best_response_p3 = all(player3_cost <= cost_matrix[i, j, z][2] for z in range(num_depth))
                    if is_best_response_p1 and is_best_response_p2 and is_best_response_p3:
                        social_cost = player1_cost + player2_cost + player3_cost
                        NEs.append(((i, j, k), social_cost))
        if not NEs:
            assert False, "No Nash Equilibrium found"
            return None
        # Return the NE with the lowest social cost
        return min(NEs, key=lambda x: x[1])[0]
    
    def _get_ne_potential_game(self, cost_matrix, potential_function_matrix):
        NEs = []
        num_rows = len(potential_function_matrix)
        num_cols = len(potential_function_matrix[0]) if num_rows > 0 else 0
        for i in range(num_rows):
            for j in range(num_cols):
                # Check if current (i, j) is a local minimum in the potential function
                local_min = True
                # Check all neighboring entries in the potential matrix
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        # Ensure indices are within the bounds of the matrix
                        if 0 <= ni < num_rows and 0 <= nj < num_cols:
                            if potential_function_matrix[ni][nj] < potential_function_matrix[i][j]:
                                local_min = False
                                break
                    if not local_min:
                        break
                if local_min:
                    NEs.append(((i, j), sum(cost_matrix[i][j])))
        return min(NEs, key=lambda x: x[1])[0]

