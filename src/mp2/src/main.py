import rospy
import numpy as np
import argparse

from gazebo_msgs.msg import  ModelState
from controller import vehicleController
import time
from waypoint_list import WayPoints
from util import euler_to_quaternion, quaternion_to_euler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_model():
    rospy.init_node("model_dynamics")
    controller = vehicleController()

    waypoints = WayPoints()
    pos_list = waypoints.getWayPoints()
    pos_idx = 1

    target_x, target_y = pos_list[pos_idx]

    def shutdown():
        """Stop the car when this ROS node shuts down"""
        # # pos
        # pos_array = np.array(controller.position) # can slice
        # waypoint_array = np.array(pos_list)
        # plt.figure()
        # plt.plot(pos_array[:,0], pos_array[:,1], 'ro', label='Position')
        # plt.plot(waypoint_array[:,0], waypoint_array[:,1], 'bo', label='Waypoints')
        # plt.scatter(0, -98, color='green', s=100, label='initial position')  
        # plt.xlabel('X Position')
        # plt.ylabel('Y Position')
        # plt.title('Vehicle Position, Waypoints')
        # plt.legend()
        # plt.savefig('position_plot.pdf')
        # plt.close()

        # # vel
        # time_list = np.linspace(0, len(controller.velocity) / 100, len(controller.velocity))  # Assuming 100 Hz rate
        # plt.figure()
        # plt.plot(time_list, controller.velocity, label='Velocity')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Velocity (m/s)')
        # plt.title('Velocity over Time')
        # plt.legend()
        # plt.savefig('velocity_plot.pdf')
        # plt.close()

        # # accel
        # plt.figure()
        # plt.plot(time_list, controller.acceleration, label='Acceleration', color='red')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Acceleration (m/s^2)')
        # plt.title('Acceleration over Time')
        # plt.legend()
        # plt.savefig('acceleration_plot.pdf')
        # plt.close()

        controller.stop()
        rospy.loginfo("Stop the car")

    rospy.on_shutdown(shutdown)

    rate = rospy.Rate(100)  # 100 Hz
    rospy.sleep(0.0)
    start_time = rospy.Time.now()
    prev_wp_time = start_time
    

    while not rospy.is_shutdown():
        rate.sleep()  # Wait a while before trying to get a new state

        # Get the current position and orientation of the vehicle
        currState =  controller.getModelState()
        
        if not currState.success:
            continue

        # Compute relative position between vehicle and waypoints
        distToTargetX = abs(target_x - currState.pose.position.x)
        distToTargetY = abs(target_y - currState.pose.position.y)

        cur_time = rospy.Time.now()
        if (cur_time - prev_wp_time).to_sec() > 4:
            print(f"failure to reach {pos_idx}-th waypoint in time")
            return False, pos_idx, (cur_time - start_time).to_sec()

        if (distToTargetX < 2 and distToTargetY < 2):
            # If the vehicle is close to the waypoint, move to the next waypoint
            prev_pos_idx = pos_idx
            pos_idx = pos_idx+1

            if pos_idx == len(pos_list): #Reached all the waypoints
                print("Reached all the waypoints")
                total_time = (cur_time - start_time).to_sec()
                print(total_time)
                return True, pos_idx, total_time

            target_x, target_y = pos_list[pos_idx]

            time_taken = (cur_time- prev_wp_time).to_sec()
            prev_wp_time = cur_time
            print(f"Time Taken: {round(time_taken, 2)}", "reached",pos_list[prev_pos_idx][0],pos_list[prev_pos_idx][1],"next",pos_list[pos_idx][0],pos_list[pos_idx][1])

        controller.execute(currState, [target_x, target_y], pos_list[pos_idx:])
if __name__ == "__main__":
    try:
        status, num_waypoints, time_taken = run_model()
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down")
