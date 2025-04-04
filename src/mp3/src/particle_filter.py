import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode

import time # debugging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import random

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1]
    curr_theta = vars[2]
    # curr_x += 0.5 * np.random.normal(0, 1)
    # curr_y += 0.5 * np.random.normal(0, 1)
    # curr_theta += 0.5 * np.random.normal(0, 1)
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        particles = list()

        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):

            # (Default) The whole map
            # x = np.random.uniform(0, world.width)
            # y = np.random.uniform(0, world.height)

            # first quadrant
            x = np.random.uniform(world.width/2, world.width)
            y = np.random.uniform(world.height/2, world.height)

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

        ###############

        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        return

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """

        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            # print(f'x1: {tmp1.size}, x2: {tmp2.size}')
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####
        if readings_robot is None:
            # If LiDAR failed, assign equal weight
            for p in self.particles:
                p.weight = 1.0 / self.num_particles
            return

        weights = []
        for p in self.particles:
            readings_particle = p.read_sensor()
            weight = self.weight_gaussian_kernel(readings_robot, readings_particle)
            p.weight = weight
            weights.append(weight)

        # Normalize weights
        total = sum(weights)
        if total == 0:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles
        else:
            for p in self.particles:
                p.weight /= total

        ###############
        # pass

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        
        weights = [p.weight for p in self.particles]
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # prevent rounding issues
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.num_particles))

        particles_new = []
        for idx in indexes:
            p = self.particles[idx]
            new_p = Particle(x=p.x, y=p.y, heading=p.heading, maze=self.world,
                            sensor_limit=self.sensor_limit, noisy=True)
            particles_new.append(new_p)

        self.particles = particles_new
        

        ###############

        self.particles = particles_new

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
            You can either use ode function or vehicle_dynamics function provided above
        """
        ## TODO #####
        
        if not self.control:
            return

        controls = self.control.copy()
        self.control = []  # reset after copying

        dt = 0.01  # time step
        for p in self.particles:
            state = [p.x, p.y, p.heading]
            solver = ode(vehicle_dynamics).set_integrator('dopri5')
            solver.set_initial_value(state, 0)

            for control_input in controls:
                v, delta = control_input
                solver.set_f_params(v, delta)
                solver.integrate(solver.t + dt)

            # Update particle with final position
            p.x = solver.y[0]
            p.y = solver.y[1]
            p.heading = solver.y[2] % (2 * np.pi)

            p.fix_invalid_particles()
        

        ###############
        # pass


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        count = 0 
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            count = 0
            errors = []

            while not rospy.is_shutdown():
                self.world.clear_objects()
                self.particleMotionModel()
                reading = self.bob.read_sensor()
                self.updateWeight(reading)
                self.resampleParticle()

                self.world.show_robot(self.bob)
                self.world.show_particles(self.particles)
                est = self.world.show_estimated_location(self.particles)

                if est:
                    err_pos = np.linalg.norm([self.bob.x - est[0], self.bob.y - est[1]])
                    err_heading = abs((self.bob.heading - est[2]) % (2 * np.pi))
                    errors.append((err_pos, err_heading))

                count += 1
                rospy.sleep(0.1)  # Control the loop rate

            ###############

