import numpy as np
import math
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
STOP_THRESHOLD = 0.02
STOP_COUNTS = 10
class BehaviouralPlanner:
    def __init__(self, lookahead, stopsign_fences, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._stopsign_fences               = stopsign_fences
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        if self._state == FOLLOW_LANE:
            closest_len, closest_index = get_closest_index(waypoints, ego_state)
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            updated_index, stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)
            self._goal_index = updated_index if stop_sign_found else goal_index
            self._goal_state = waypoints[self._goal_index]
            if stop_sign_found:
                self._goal_state[2] = 0
                self._state = DECELERATE_TO_STOP
            pass
        elif self._state == DECELERATE_TO_STOP:
            self._state = STAY_STOPPED if closed_loop_speed < STOP_THRESHOLD else DECELERATE_TO_STOP
            pass
        elif self._state == STAY_STOPPED:
            if self._stop_count == STOP_COUNTS:
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                _, stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                if not stop_sign_found:
                    self._stop_count = 0
                    self._state = FOLLOW_LANE
                pass
            else:
                self._stop_count += 1
                pass
        else:
            raise ValueError('Invalid state value.')
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        arc_length = closest_len
        wp_index = closest_index
        if arc_length > self._lookahead:
            return wp_index
        if wp_index == len(waypoints) - 1:
            return wp_index
        num_waypoints = len(waypoints)
        for i in range(wp_index + 1, num_waypoints):
            arc_length += math.sqrt(
                (waypoints[i][0] - waypoints[i-1][0])**2 + (waypoints[i][1] - waypoints[i-1][1])**2
            )
            if arc_length > self._lookahead:
                break
        return i
    def check_for_stop_signs(self, waypoints, closest_index, goal_index):
        for i in range(closest_index, goal_index):
            intersect_flag = False
            for stopsign_fence in self._stopsign_fences:
                wp_1   = np.array(waypoints[i][0:2])
                wp_2   = np.array(waypoints[i+1][0:2])
                s_1    = np.array(stopsign_fence[0:2])
                s_2    = np.array(stopsign_fence[2:4])

                v1     = np.subtract(wp_2, wp_1)
                v2     = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1     = np.subtract(s_2, s_1)
                v2     = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True
                if intersect_flag:
                    goal_index = i
                    return goal_index, True
        return goal_index, False
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        if not self._follow_lead_vehicle:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return
            lead_car_delta_vector = np.divide(lead_car_delta_vector,
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]),
                                  math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector,
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return
            self._follow_lead_vehicle = True
        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return
            self._follow_lead_vehicle = False
def get_closest_index(waypoints, ego_state):
    closest_len = float('Inf')
    closest_index = 0
    for i, wp in enumerate(waypoints):
        d = math.sqrt((ego_state[0] - wp[0])**2 + (ego_state[1] - wp[1])**2)
        if d < closest_len:
            closest_len = d
            closest_index = i
    return closest_len, closest_index
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
