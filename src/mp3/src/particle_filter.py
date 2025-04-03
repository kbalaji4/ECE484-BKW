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

        # get total weight: get sensor reading from each particle and compare with real robot sensor readings
        total_weight = 0
        for particle in self.particles:
            readings_particle = particle.read_sensor() # this reads 4 or 8 directions
            # robot reads 4, particle reads 8. why particle read 8
            # readings robot is first cuz it's x1 in weight_gaussian_kernel. either gauss or uniform
            particle.weight = self.weight_gaussian_kernel(readings_robot, readings_particle)
            total_weight += particle.weight
        
        # normalize weight
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight

        else:
            print(f'total weight: {total_weight}, setting uniform weights')
            # if total weight is 0, set all weights to be uniform. would this ever happen tho?
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)
        ###############

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

        # 1: array of cumsum
        weights = [particle.weight for particle in self.particles] # weights are normalized [0, 1]
        cumulative_sum = np.cumsum(weights)

        """ 
        randomness
        """ 
        # num_random_particles = 0
        num_random_particles = int(0.0025 * self.num_particles)  #  randomness
        for _ in range(num_random_particles):
            randx = np.random.uniform(0, self.world.width)
            randy = np.random.uniform(0, self.world.height)
            # noisy true. default heading is rando sampled
            particles_new.append(Particle(
                x=randx,
                y=randy,
                maze=self.world,
                sensor_limit=self.sensor_limit,
                noisy=True,
                weight = 1.0/self.num_particles # uniform default
            ))

        for _ in range(self.num_particles - num_random_particles):
            # 2: rando sample from cumsum
            random_number = np.random.uniform(0, 1) # sample from cumsum basically
            index = np.searchsorted(cumulative_sum, random_number)
            # 3: index corresponds to particle we want to create        self.particles = particles_new

            selected_particle = self.particles[index]
            # noise = 0.05 * np.random.normal(0, 1) # noise for x and y
            # 4: create a new particle with noise. we are doing this for each particle duh
            particles_new.append(Particle(
                x=selected_particle.x,
                y=selected_particle.y,
                maze=self.world,
                heading=selected_particle.heading,
                sensor_limit=self.sensor_limit,
                noisy=True
                ,weight=selected_particle.weight
            ))
        ###############

        self.particles = particles_new

    def resampleParticleQuadrant(self):
        particles_new = []
    
        # Calculate quadrant weights
        quadrant_weights = [0, 0, 0, 0]
        quadrant_indices = [[], [], [], []]
        for idx, p in enumerate(self.particles):
            if p.x < self.world.width / 2 and p.y < self.world.height / 2:
                quadrant_indices[0].append(idx)
                quadrant_weights[0] += p.weight
            elif p.x >= self.world.width / 2 and p.y < self.world.height / 2:
                quadrant_indices[1].append(idx)
                quadrant_weights[1] += p.weight
            elif p.x < self.world.width / 2 and p.y >= self.world.height / 2:
                quadrant_indices[2].append(idx)
                quadrant_weights[2] += p.weight
            else:
                quadrant_indices[3].append(idx)
                quadrant_weights[3] += p.weight
        
        # Normalize quadrant weights
        total_q_weight = sum(quadrant_weights)
        if total_q_weight > 0:
            quadrant_probs = [w/total_q_weight for w in quadrant_weights]
        else:
            quadrant_probs = [0.25, 0.25, 0.25, 0.25]
        
        # add randomness, particles are sampled globally


        num_random_particles = int(0.01 * self.num_particles)  # 1% randomness
        for _ in range(num_random_particles):
            randx = np.random.uniform(0, self.world.width)
            randy = np.random.uniform(0, self.world.height)
            # noisy true. default heading is rando sampled
            particles_new.append(Particle(
                x=randx,
                y=randy,
                maze=self.world,
                sensor_limit=self.sensor_limit,
                noisy=True,
                weight = 1.0/self.num_particles # uniform default
            ))
        
        quadrant_cumsums = []
        for q_indices in quadrant_indices:
            q_weights = [self.particles[idx].weight for idx in q_indices]
            q_cumsum = np.cumsum(q_weights) if q_weights else []
            quadrant_cumsums.append(q_cumsum)
        # print(f'quadraunt cum sums: {quadrant_cumsums}')
        # print(f'len(quadrant_cumsums): {len(quadrant_cumsums[0]), len(quadrant_cumsums[1]), len(quadrant_cumsums[2]), len(quadrant_cumsums[3])}')


        num_quadrant_particles = self.num_particles - num_random_particles

        for _ in range(num_quadrant_particles):
            # First select a quadrant based on quadrant weights
            selected_quadrant = np.random.choice(4, p=quadrant_probs)
            
            # Then resample from particles in that quadrant
            if quadrant_indices[selected_quadrant]:
                random_number = np.random.uniform(0, 1)
                selected_idx_within_quadrant = np.searchsorted(quadrant_cumsums[selected_quadrant], random_number) - 1 # - 1? idk
                # print(f'selected idx within quadrant: {selected_idx_within_quadrant}')
                selected_idx = quadrant_indices[selected_quadrant][selected_idx_within_quadrant]
                selected_particle = self.particles[selected_idx]
                # don't just do random.choice lol
            else:
                # if no particles in the quadrant we picked, sample
                
                # just global random sample lol
                # selected_particle = random.choice(self.particles)

                # below is multinomial sampling
                selected_particle = self.particles[np.searchsorted(cumulative_sum, np.random.uniform(0, 1))]
            
            particles_new.append(Particle(
                x=selected_particle.x,
                y=selected_particle.y,
                maze=self.world,
                heading=selected_particle.heading,
                sensor_limit=self.sensor_limit,
                noisy=True
                ,weight=selected_particle.weight
            ))
        
        self.particles = particles_new
    
    def resampleParticleSystematic(self):
        """
        Description:
            Perform systematic resampling to get a new list of particles.
            Uses a single random number u_bar ~ U[0,1) to generate ordered
            numbers u_k = (k-1 + u_bar)/N for selecting particles.
        """
        particles_new = list()
        weights = np.array([particle.weight for particle in self.particles])
        N = len(self.particles)
        
        # Generate one random number u_bar ~ U[0,1)
        u_bar = np.random.uniform(0, 1)
        
        # Generate systematic points u_k = (k-1 + u_bar)/N
        u_k = (np.arange(N) + u_bar) / N
        
        # Calculate cumulative sum of weights
        cumsum = np.cumsum(weights)
        
        # Initialize index for original particles
        index = 0
        
        # Loop through systematic points
        for u in u_k:
            # Find the particle index for this systematic point
            while cumsum[index] < u:
                index += 1
                
            # Get the selected particle
            selected_particle = self.particles[index]
            
            # Create new particle with noise
            particles_new.append(Particle(
                x=selected_particle.x,
                y=selected_particle.y,
                maze=self.world,
                heading=selected_particle.heading,
                sensor_limit=self.sensor_limit,
                noisy=True,
                weight=selected_particle.weight
            ))
        
        self.particles = particles_new
    def resampleParticleStratified(self):
        """
        Description:
            Perform stratified resampling to get a new list of particles.
            Each stratum has size 1/N and a random number is drawn from within each stratum.
        """
        particles_new = list()
        weights = np.array([particle.weight for particle in self.particles])
        N = len(self.particles)
        
        # Generate stratified random numbers
        # Each random number is drawn from its own stratum
        random_numbers = (np.arange(N) + np.random.uniform(0, 1, N)) / N
        
        # Calculate cumulative sum of weights
        cumsum = np.cumsum(weights)
        
        # Initialize index for original particles
        index = 0
        
        # Loop through all stratified random numbers
        for random_num in random_numbers:
            # Find the particle index for this random number
            while cumsum[index] < random_num:
                index += 1
            
            # Get the selected particle
            selected_particle = self.particles[index]
            
            # Create new particle with noise
            particles_new.append(Particle(
                x=selected_particle.x,
                y=selected_particle.y,
                maze=self.world,
                heading=selected_particle.heading,
                sensor_limit=self.sensor_limit,
                noisy=True,
                weight=selected_particle.weight
            ))
        
        self.particles = particles_new

    def resampleParticleResidual(self):
        """
        Description:
            Perform residual resampling to get a new list of particles.
            First deterministically allocate particles based on integer weights,
            then use multinomial resampling for the remainder.
        """
        particles_new = list()
        N = len(self.particles)
        weights = np.array([particle.weight for particle in self.particles])
        
        # Calculate number of copies for each particle (first step)
        N_weights = N * weights
        integer_weights = np.floor(N_weights).astype(int)
        sum_integer_weights = np.sum(integer_weights)
        
        # First pass: deterministic allocation
        for i, count in enumerate(integer_weights):
            for _ in range(count):
                selected_particle = self.particles[i]
                particles_new.append(Particle(
                    x=selected_particle.x,
                    y=selected_particle.y,
                    maze=self.world,
                    heading=selected_particle.heading,
                    sensor_limit=self.sensor_limit,
                    noisy=True,
                    weight=selected_particle.weight
                ))
        
        # Second pass: multinomial resampling for remaining particles
        remaining_N = N - sum_integer_weights
        if remaining_N > 0:
            # Calculate residual weights
            residual_weights = (N_weights - integer_weights) / remaining_N
            
            # Normalize residual weights
            residual_weights = residual_weights / np.sum(residual_weights)
            
            # Compute cumulative sum for residual weights
            cumsum = np.cumsum(residual_weights)
            
            # Sample remaining particles
            for _ in range(remaining_N):
                random_number = np.random.uniform(0, 1)
                index = np.searchsorted(cumsum, random_number)
                selected_particle = self.particles[index]
                
                particles_new.append(Particle(
                    x=selected_particle.x,
                    y=selected_particle.y,
                    maze=self.world,
                    heading=selected_particle.heading,
                    sensor_limit=self.sensor_limit,
                    noisy=True,
                    weight=selected_particle.weight
                ))
        
        self.particles = particles_new
    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot.
            You can either use ode function or vehicle_dynamics function, we first choose vehicle_dynamics
            1. for every particle
            2. get the new x, y, heading for that particle using the vehicle_dynamics function and new control
            3. do this for all the new control steps that appeared since the last update (each control signal is 0.01s)

            output: none, it just modifies self.particles and self.control
        """
        ## TODO #####
        # vehicle_dynamics(t, vars, vr, delta)
        dt = 0.01 # timestep of each control signal
        # get control signals since last update, u should keep clearing this list
        for particle in self.particles:
            cur_x, cur_y, cur_heading = particle.x, particle.y, particle.heading
            for control_step in self.control:
                v, delta = control_step
                # newx, y, heading is just the current lol, bad naming
                dtheta = delta * dt
                cur_x += v * (np.sin(cur_heading + dtheta) - np.sin(cur_heading))/delta
                cur_y += v * (np.cos(cur_heading) - np.cos(cur_heading + dtheta))/delta
                cur_heading += dtheta
                # print(f'newx, newy, newheading: {newx, newy, newheading}')
            particle.x, particle.y, particle.heading = cur_x, cur_y, cur_heading # update particle pos
            particle.fix_invalid_particles()
            # wait what does try_move() do? it's never called, is it just for us
        self.control.clear() # clear control signal list for next update
        # for particle in self.particles:
        #     newx, newy, newheading = particle.x, particle.y, particle.heading
        #     for control_step in self.control:
        #         v, delta = control_step
        #         # newx, y, heading is just the current lol, bad naming
        #         dx, dy, dtheta = vehicle_dynamics(dt, [newx, newy, newheading], v, delta)
        #         newx += dx * dt
        #         newy += dy * dt
        #         newheading += dtheta * dt
        #         # print(f'newx, newy, newheading: {newx, newy, newheading}')
        #     particle.x, particle.y, particle.heading = newx, newy, newheading # update particle pos
        #     particle.fix_invalid_particles()
        #     # wait what does try_move() do? it's never called, is it just for us
        # self.control.clear() # clear control signal list for next update
        ###############


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
        """ 
        bob has methods to getModelState, read_sensor from LidarProcessing
        bob getModelState (Robot class super of Particle) 
        """
        print(f'initial bob model state: {self.bob.getModelState()}') 
        print(f'initial bob read sensor: {self.bob.read_sensor()}')
        # print(f'initial model state: {self.getModelState()}')
        # count = 0 
        self.world.clear_objects()
        i = 0
        update_frequency = 1 
        plot_update_frequency = 1
        directionsPrinted = 0

        # error tracking
        position_errors = []
        orientation_errors = []

        # # position, orientation error plots
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
        # ax1.set_title("Position Error (Euclidean Distance)")
        # ax1.set_xlabel("Iterations")
        # ax1.set_ylabel("Error (meters)")
        # ax2.set_title("Orientation Error")
        # ax2.set_xlabel("Iterations")
        # ax2.set_ylabel("Error (radians)")

        # try plotting weight distributions? like x, y position and then weight color
        fig_weights, ax_weights = plt.subplots(figsize=(5, 4))
        ax_weights.set_title("Particle Weight Distribution")
        ax_weights.set_xlabel("Y Position")
        ax_weights.set_ylabel("X Position")
        scatter = ax_weights.scatter([], [], c=[], cmap="viridis", s=10)
        fig_weights.colorbar(scatter, ax=ax_weights, label="Particle Weight")

        # # Initialize plots for sensor readings and particle averages
        # fig, (ax_sensor, ax_particle_avg) = plt.subplots(2, 1, figsize=(6, 5))
        # ax_sensor.set_title("Robot Sensor Readings")
        # ax_sensor.set_xlabel("Direction")
        # ax_sensor.set_ylabel("Distance (cm)")
        # ax_sensor.set_xticks([])  # Will set dynamically based on direction count

        # ax_particle_avg.set_title("Average Particle Sensor Readings")
        # ax_particle_avg.set_xlabel("Direction")
        # ax_particle_avg.set_ylabel("Distance (cm)")
        # ax_particle_avg.set_xticks([])  # Will set dynamically based on direction count


        # update 1/update_frequency times
        while True:
            readings = self.bob.read_sensor()
            if readings and directionsPrinted == 0:
                directionsPrinted = 1
                print(f'num directions: {len(readings), readings}')
            self.world.clear_objects() # super necessary, otherwise u get streaks
            # time.sleep(0.50) # may not be necessary
            """ 
            read_sensor alr updates x, y, heading for you
            """
            if i % update_frequency == 0:
                self.particleMotionModel()  # Predict particle states, "sample motion model"
                readings_robot = self.bob.read_sensor() # get the actual readings, alr converted to the gazebo?
                self.updateWeight(readings_robot)  # Update particle weights
                self.resampleParticle()  # Resample particles, this updates self.particles in place alr
                # self.resampleParticleQuadrant() # quadrant resample, quite fast but wrong?
                # self.resampleParticleSystematic()
                # self.resampleParticleStratified()
                # self.resampleParticleResidual()

            self.world.show_robot(self.bob)
            self.world.show_particles(self.particles, show_frequency = 10)
            estimated_location = self.world.show_estimated_location(self.particles) # estimated?

            # plot time
            # if estimated_location:
            #     x_est, y_est, heading_est = estimated_location
            #     x_actual, y_actual = self.bob.x, self.bob.y
            #     heading_actual = self.bob.heading

            #     position_error = np.sqrt((x_est - x_actual) ** 2 + (y_est - y_actual) ** 2)
            #     orientation_error = abs(heading_est - heading_actual)

            #     position_errors.append(position_error)
            #     orientation_errors.append(orientation_error)
            if i % plot_update_frequency == 0:
                # # Update error plots, ok maybe not live then
                # ax1.clear()
                # ax1.plot(position_errors, label="Position Error")
                # ax1.legend()

                # ax2.clear()
                # ax2.plot(orientation_errors, label="Orientation Error")
                # ax2.legend()

                
                # weight distribution, don't rely on estimated_location but keep them in frequency
                ax_weights.clear()
                x_positions = [particle.x for particle in self.particles]
                y_positions = [particle.y for particle in self.particles]
                weights = [particle.weight for particle in self.particles]
                scatter = ax_weights.scatter(x_positions, y_positions, c=weights, cmap="viridis", s=10)
                # plt.pause(0.01)

                # distance plots
                # if readings_robot:
                #     # Plot robot sensor readings
                #     ax_sensor.clear()
                #     directions = ['front', 'right', 'rear', 'left', 'front_left', 'front_right', 'rear_left', 'rear_right']
                #     # directions = [f"Dir {j+1}" for j in range(len(readings_robot))]
                #     ax_sensor.set_xticks(range(len(readings_robot)))
                #     ax_sensor.set_xticklabels(directions)
                #     ax_sensor.bar(range(len(readings_robot)), readings_robot, color="blue", alpha=0.7)

                #     # Calculate and plot average particle readings
                #     particle_readings = [particle.read_sensor() for particle in self.particles]
                #     avg_particle_readings = np.mean(particle_readings, axis=0) if particle_readings else []
                #     ax_particle_avg.clear()
                #     ax_particle_avg.set_xticks(range(len(avg_particle_readings)))
                #     ax_particle_avg.set_xticklabels(directions)
                #     ax_particle_avg.bar(range(len(avg_particle_readings)), avg_particle_readings, color="orange", alpha=0.7)
                plt.pause(0.01)
            i+= 1
                ###############
