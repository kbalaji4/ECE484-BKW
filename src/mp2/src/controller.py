import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True
        self.acceleration = []
        self.position = []
        self.velocity = []

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y= currentPose.pose.position.x, currentPose.pose.position.y
        vel = currentPose.twist.linear # vector?
        yaw = quaternion_to_euler(currentPose.pose.orientation.x, currentPose.pose.orientation.y, currentPose.pose.orientation.z, currentPose.pose.orientation.w)[2]
        # the helper just gets it for you lol
        vel_scalar = np.sqrt(vel.x**2 + vel.y**2)
        # print("vel: ", vel, vel_scalar)
        # print("pos: ", pos_x, pos_y)
        # print("yaw: ", yaw)
        # so we return xy position, scalar velocity, and like orientation

        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel_scalar, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################
        straight_target_velocity = 20
        curve_target_velocity = 18

        target_velocity = 0

        # arbitrarily pick 10 waypoints ahead. if x is straight or y is straight, try to reach straight_target_velocity.
        # ig if turn then both are curving lol

        num_waypoints_ahead = 5
        """
        angle threshold: if angle between two waypoints is greater than this, then start turning
        """
        angle_threshold = 0.5 # 0.5 rad = 28.6479 degrees

        # not many
        if len(future_unreached_waypoints) < num_waypoints_ahead:
            num_waypoints_ahead = len(future_unreached_waypoints)
        
        angles = []
        for i in range(num_waypoints_ahead - 1):
            x1, y1 = future_unreached_waypoints[i]
            x2, y2 = future_unreached_waypoints[i + 1]
            angle = math.atan2(y2 - y1, x2 - x1)
            angles.append(angle)
        
        angle_differences = [abs(angles[i+1] - angles[i]) for i in range(len(angles) - 1)]
        
        # is path straight or curved. adjust velocity otherwise
        if all(difference < math.radians(angle_threshold) for difference in angle_differences):
            target_velocity = straight_target_velocity
        else:
            target_velocity = curve_target_velocity
        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity


    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        k = 0.5
        min_look = 1.5
        
        state = self.getModelState()
        if state is None:
            return 0.0

        v_x = state.twist.linear.x
        v_y = state.twist.linear.y
        speed = math.sqrt(v_x ** 2 + v_y ** 2)
        l_d = min_look + k * speed
        lookahead_point = None
        for waypoint in future_unreached_waypoints:
            dist = math.sqrt((waypoint[0] - curr_x) ** 2 + (waypoint[1] - curr_y) ** 2)
            if dist >= l_d:
                lookahead_point = waypoint
                break
        if lookahead_point is None:
            lookahead_point = future_unreached_waypoints[-1]
            
        dx = lookahead_point[0] - curr_x
        dy = lookahead_point[1] - curr_y
        ld = math.sqrt(dx ** 2 + dy ** 2)
        alpha = math.atan2(dy, dx) - curr_yaw
        delta = math.atan2(2 * self.L * math.sin(alpha), ld)
        return delta
    
    def pure_pursuit_lateral_controller_OG(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        L = self.L  # Wheelbase
        lookahead_distance = 10  

        # Calculate the lookahead point
        target_x, target_y = target_point

        # Calculate the angle to the target point
        dx = target_x - curr_x
        dy = target_y - curr_y
        alpha = math.atan2(dy, dx) - curr_yaw

        # Normalize alpha to be within the range [-pi, pi]
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        # Calculate the steering angle using the Pure Pursuit formula
        delta = math.atan2(2 * L * math.sin(alpha), lookahead_distance)

        return delta


    def execute(self, currentPose, target_point, future_unreached_waypoints, position_list, velocity_list, acceleration_list):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) # apparently ours is not. 
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz
            # print(curr_x, curr_y, curr_vel, curr_yaw)
            self.acceleration.append(acceleration)
            self.prev_vel = curr_vel # oh update
        self.velocity.append(curr_vel)
        self.position.append([curr_x, curr_y])


        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)
