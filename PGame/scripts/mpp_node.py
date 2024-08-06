import rospy
import numpy as np
import time

from motion_primitive_planner.mpp import MPP
from gtm_planner.utils.guideline import create_laneletmap, get_guidelines
import os
import rospkg
from vehicle_msgs.srv import GetAction, GetActionResponse, Reset    
from vehicle_msgs.msg import State
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class MPPNode:
    def __init__(self, lane_dict):
        self._lane_dict = lane_dict
        self._vehicle_states = {}
        self._mpp = MPP(self._lane_dict, self._vehicle_states)

        self._get_action_srv = rospy.Service('/motion_planner/get_ev_action', GetAction, self._get_action_callback)
        self._reset_srv = rospy.Service('/potential_game/reset', Reset, self.reset)
        self._waypoints_pub = rospy.Publisher("/visualizer/waypoints", Marker, queue_size=10)

    def reset(self, req):
        self._desired_velocities = {}
        self._averaged_desired_velocities = {}
        self._mpp.reset()   
        return True
    
    @staticmethod
    def average1(vel, av_vel_pre, vel_des):
        if av_vel_pre == None:
            av_vel = 0.8*vel + 0.2*vel_des
        else:
            av_vel = 0.6*vel + 0.4*av_vel_pre            
            # av_vel = 0.7*av_vel + 0.3*vel_des        
            av_vel = 0.3*av_vel + 0.7*vel_des             
        return av_vel
    
    @staticmethod
    def average2(vel, av_vel_pre, vel_des):
        if av_vel_pre == None:
            av_vel = 0.9*vel + 0.1*vel_des
        else:
            av_vel = 0.6*vel + 0.4*av_vel_pre            
            av_vel = 0.9*av_vel + 0.1*vel_des
                        
        return av_vel

    def _get_action_callback(self, req):
        num_vehicles = len(req.veh_states)
        inter_axle_lengths = [req.veh_states[i].wheel_base for i in range(num_vehicles)]
        vehicle_lengths = [req.veh_states[i].length for i in range(num_vehicles)]
        vehicle_widths = [req.veh_states[i].width for i in range(num_vehicles)]
        
        for i in range(num_vehicles):
            heading = req.veh_states[i].heading
            self._vehicle_states[i] = [
                req.veh_states[i].x - inter_axle_lengths[i] * np.cos(heading) / 2.0,
                req.veh_states[i].y - inter_axle_lengths[i] * np.sin(heading) / 2.0,
                heading,
                req.veh_states[i].vel,
                req.veh_states[i].acc
            ]
            self._desired_velocities[i] = req.veh_states[i].vel_des
            
        if len(self._averaged_desired_velocities) == 0:
            for key in self._desired_velocities:
                vel = self._vehicle_states[key][3]
                if key == 0:
                    self._averaged_desired_velocities[key] = self.average1(vel, None, self._desired_velocities[key])
                else:
                    self._averaged_desired_velocities[key] = self.average2(vel, None, self._desired_velocities[key])
        else:
            for key in self._desired_velocities:
                vel = self._vehicle_states[key][3]
                if key == 0:
                    self._averaged_desired_velocities[key] = self.average1(vel, self._averaged_desired_velocities[key], self._desired_velocities[key])
                else:
                    self._averaged_desired_velocities[key] = self.average2(vel, self._averaged_desired_velocities[key], self._desired_velocities[key])
            
        
        rospy.loginfo("desired_velocities: %s", self._desired_velocities)
        rospy.loginfo("averaged_desired_velocities: %s", self._averaged_desired_velocities)
        start_time = time.time()
        acc, steer, best_trajectory = self._mpp.run_step(inter_axle_lengths, 
                                                         self._averaged_desired_velocities, 
                                                         vehicle_lengths, vehicle_widths)
        computation_time = time.time() - start_time
        rospy.loginfo("Total time for planning: %.2f s", computation_time)
        print('\n')
        if best_trajectory is not None:
            self.visualize(best_trajectory, inter_axle_lengths[0])
            next_ego_state = State()
            next_ego_state.x = best_trajectory.x[1]
            next_ego_state.y = best_trajectory.y[1]
            next_ego_state.heading = best_trajectory.heading[1]
            next_ego_state.vel = best_trajectory.vel[1]
            next_ego_state.acc = best_trajectory.acc_long[1]
            next_ego_state.steer = best_trajectory.steer[1]
        next_ego_state = State()
        # print(f"x_traj:\n{best_trajectory.x}")
        # print(f"y_traj:\n{best_trajectory.y}")
        # print(f"d_traj:\n{best_trajectory.d}")
        # print(f"d_dot_traj:\n{best_trajectory.d_d}")
        
        # print(f"vel:\n{best_trajectory.vel}")
        # # print(f"acc:\n{best_trajectory.acc_long}")
        # print(f"steer:\n{best_trajectory.steer}")
        # print(f"heading:\n{best_trajectory.heading}")
        # print(f"\n\n")
        
        # next_ego_state.x = best_trajectory.x[2]
        # next_ego_state.y = best_trajectory.y[2]
        # next_ego_state.heading = best_trajectory.heading[2]
        # next_ego_state.vel = best_trajectory.vel[2]
        # next_ego_state.acc = best_trajectory.acc_long[2]
        # next_ego_state.steer = best_trajectory.steer[2]
        return GetActionResponse(acc, steer, computation_time, next_ego_state)

    def visualize(self, traj, inter_axle_length):
        waypoints_marker = Marker()
        waypoints_marker.id = 0;
        waypoints_marker.type = Marker.SPHERE_LIST;
        waypoints_marker.header.stamp = rospy.Time.now();
        waypoints_marker.header.frame_id = "map";
        waypoints_marker.pose.orientation.w = 1.00;
        waypoints_marker.action = Marker.ADD;
        waypoints_marker.ns = "waypoints";
        waypoints_marker.color.r = 0.36;
        waypoints_marker.color.g = 0.74;
        waypoints_marker.color.b = 0.89;
        waypoints_marker.color.a = 0.50;
        waypoints_marker.scale.x = 0.5;
        waypoints_marker.scale.y = 0.5;
        waypoints_marker.scale.z = 0.5;
        
        for x, y, heading in zip(traj.x, traj.y, traj.heading):
            point = Point()
            point.x = x + inter_axle_length * np.cos(heading) / 2.0
            point.y = y + inter_axle_length * np.sin(heading) / 2.0
            point.z = 0.2
            waypoints_marker.points.append(point)
        
        self._waypoints_pub.publish(waypoints_marker)


if __name__ == '__main__':
    rospy.init_node('mpp_node')
    root_dir = os.path.dirname(rospkg.RosPack().get_path('sim_env'))
    maps_dir = os.path.join(root_dir, "maps")    
    scenario_name = 'DR_CHN_Merging_ZS0'
    lanelet_map_ending = ".osm"
    lanelet_map_file = os.path.join(maps_dir, scenario_name + lanelet_map_ending)
    laneletmap =  create_laneletmap(lanelet_map_file)
    current_guideline, target_guideline = get_guidelines(laneletmap)
    lane_dict = {'current_guideline': current_guideline.get_waypoints(),
                 'target_guideline': target_guideline.get_waypoints()}
    mpp_node = MPPNode(lane_dict)
    rospy.spin()
