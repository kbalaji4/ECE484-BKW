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

import random

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
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
            x = np.random.uniform(0, world.width)
            y = np.random.uniform(0, world.height)


            ## first quadrant
            # x = 
            # y =

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
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))


    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """

        ## TODO #####

        ###############
        # pass

    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()

        ## TODO #####
        """ 
        u can construct particles with Particle(...noisy=True)
        recommended to use multinomial resampling for this method

        1. Calculate an array of the cumulative sum of the weights.
        2. Randomly generate a number and determine which range in that cumulative weight array to which
        the number belongs.        ## TODO: Add 4 additional sensor directions #####

        3. The index of that range would correspond to the particle that should be created.
        4. Repeat sampling until you have the desired number of samples.
        """
        

        ###############

        self.particles = particles_new

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
            You can either use ode function or vehicle_dynamics function provided above
        """
        ## TODO #####
        # vehicle_dynamics(t, vars, vr, delta)
        

        ###############
        # pass


    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        print(f'num_particles {self.num_particles}')
        print(f'sensor_limit {self.sensor_limit}')
        print(f'initial particles: {len(self.particles)}')
        print(f'x start: {self.x_start}') # gazebo or python?
        print(f'y start: {self.y_start}')
        print(f'bob: {self.bob}')
        # print(f'world: {self.world}')
        print(f'initial model state: {self.getModelState()}')
        count = 0 
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            # modelState = self.GetModelState() # how do u get current particles?

            # sample motion model (p) which are teh particles representing the current distribution
            """ 
            self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
            self.control = []                   # A list of control signal from the vehicle

            time step is 0.01s. control is an append only log of the actual v, delta control signals
            vehicle dynamics can take these v, delta as inputs and output x, y, theta (orientation)

             You should perform integra-
        tion through the whole list of control input stored in self.control with time step 0.01. Since the vehicle
        control frequency is higher than the particle update frequency, a list of vehicle control inputs will be stored.
        Therefore, if you only use the most recent control input, your particle motion will be wrong. By doing this,
        you can properly predict the new location of the particle.
            """
            # print(f'count: {count}')
            # print(f'model state: {self.getModelState()}')
            # print(f'control: {len(self.control)}')
            # count += 1
            # time.sleep(1)
            # current_state = []
            # vehicle_dynamics(t, vars, vr, delta):
            #     curr_x = vars[0]
            #     curr_y = vars[1] 
            #     curr_theta = vars[2]
                
            #     dx = vr * np.cos(curr_theta)
            #     dy = vr * np.sin(curr_theta)
            #     dtheta = delta
            #     return [dx,dy,dtheta]

            # reading = vehicl_read_sensor() # this might be model state
            # self.updateWeight(readings_robot) # updateWeight(p, reading) 

            # self.resampleParticle() # p = resampleParticle(p)
               
                ###############
