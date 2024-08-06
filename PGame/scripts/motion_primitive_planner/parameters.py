import math

desired_velocities = {"EV": 10, "SV0": 10, "SV1": 10, "SV2": 10}
target_lanes = {"EV": 3.5, "SV0": 3.5, "SV1": 3.5, "SV2": 3.5}    
lane_width = 3.5    

T = 5
dt = 0.1
MIN_T = 5.0
MAX_T = 5.0
DT = 0.5

# Typical dimensions for a Tesla Model 3
# vehicle_length = 4.694  # in meters
# vehicle_width = 1.849  # in meters

# Parameters for the trajectory cost
all_cost_weights = {
    "proximity": 5,
    0: {
        "jerk": 1,
        "efficiency": 1,
        "target_lane": 10,
    },
    1: {
        "jerk": 1,
        "efficiency": 10,
        "target_lane": 1,
    },
    2: {
        "jerk": 1,
        "efficiency": 10,
        "target_lane": 1,
    },
    
}

# Parameters for the motion primitive planner
MAX_LAT_DIST = 0.0
MIN_LAT_DIST = -4.0
D_LAT_DIST = 1.0

MAX_SPEED = 6.0
MIN_SPEED = 0.5
D_V = 0.5

collision_check_inflation = 1

if __name__ == "__main__":
    import numpy as np
    num_action = 0
    for v_terminal in np.arange(MIN_SPEED, MAX_SPEED + 0.5*D_V, D_V):
        for y_terminal in np.arange(MIN_LAT_DIST, MAX_LAT_DIST + D_LAT_DIST, D_LAT_DIST):
            print(f"Action {num_action}: v_terminal = {v_terminal}, y_terminal = {y_terminal}")
            num_action += 1
    print("\n")
    num_action = 0
    for v_terminal in np.arange(MIN_SPEED, MAX_SPEED + 0.5*D_V, D_V):
        print(f"Action {num_action}: v_terminal = {v_terminal}")
        num_action += 1