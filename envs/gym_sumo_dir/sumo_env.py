import gym
import traci
import sumolib
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
import os
from gym import spaces
from gym.spaces import Discrete, MultiDiscrete

# reference-----https://github.com/Ilhem23/change_lane_DQN

def angle_between(a1, a2, angle_rl):
	x_diff = a2[0] - a1[0]
	y_diff = a2[1] - a1[1]
	ang = degrees(atan2(y_diff, x_diff))
	ang += angle_rl
	ang = ang % 360
	return ang


def get_distance(x, y):
	return distance.euclidean(x, y)


class SumoEnv(gym.Env):
	def __init__(self):
		self.Id = 'DQN-Agent'
		self.action_space = Discrete(3)
		self.observation_space = spaces.Box(low= np.zeros((6, 5)), high = np.full((6, 5), float('inf')), shape = (6, 5), dtype = np.float32) 
		self.step_length = 0.4
		self.acceleration_records = deque([0, 0], maxlen=2)
		self.obs_dim = 3
		self.state_dim = (4*self.obs_dim*self.obs_dim)+1
		self.present_lane = ''
		self.present_sublane = -1
		self.max_speed = 0
		self.speed = 0
		self.lateral_speed = 0
		self.acceleration = 0
		self.angle = 0
		self.gui = False
		self.no = 0
		self.veh_type = 0
		self.lane_id_list = []
		self.max_steps = 10000
		self.present_step = 0
		self.collision = False
		self.terminal = False

	def start(self, gui=False, no_Vehicle=30, veh_type='vehicle'):
		self.gui = gui
		self.no_Vehicle = no_Vehicle
		self.veh_type = veh_type
		#self.network_conf = network_conf
		#self.net = sumolib.net.readNet(network_xml)
		self.present_step = 0
		self.collision = False
		self.terminal = False

		if self.gui:
			#sumoBinary = r"\Users\vyshnavi\Applications\SUMO-GUI.app"
			sumoBinary = sumolib.checkBinary('sumo-gui')
			sumoCmd = [sumoBinary, "-c", r"/Users/vyshnavi/Documents/custom-gym/offramps.sumocfg"]
			traci.start(sumoCmd) # opens and runs the sumoconfig file
		else:
			#sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"
			#sumoCmd = [sumoBinary, "-c", r"C:\Users\vyshn\Desktop\custom-gym\offramps.sumocfg"]
			traci.load(['-c',r"/Users/vyshnavi/Documents/custom-gym/offramps.sumocfg"])
			print('Loading called')

		self.lane_id_list = traci.lane.getIDList()

		# Populating the highway

		for i in range(self.no_Vehicle):
			veh_Id = 'vehicle_' + str(i)
			traci.vehicle.add(veh_Id, routeID='route_1', typeID=self.veh_type, departLane='random')
			lane_change_model = 256
			traci.vehicle.setLaneChangeMode(veh_Id, lane_change_model)
		traci.vehicle.add(self.Id, routeID='route_1', typeID='rl')

		# Do some random step to distribute the vehicles
		for step in range(self.no_Vehicle*4):
			traci.simulationStep()

		# Setting the lane change mode to 0 meaning that we disable any autonomous lane change and collision avoidance
		traci.vehicle.setLaneChangeMode(self.Id, 0)
		traci.vehicle.setSpeedMode(self.Id,0)

		# Setting up useful parameters
		self.update_params()


	def update_params(self):
		#print(self.compute_observation_matrix())
		# initialize params
		try:
			self.pos = traci.vehicle.getPosition(self.Id)
			print('Position:',self.pos)
			self.present_lane = traci.vehicle.getLaneID(self.Id)
			print('current lane: ',self.present_lane)
			if self.present_lane == '':
				assert self.collision
				while self.Id in traci.simulation.getStartingTeleportIDList() or traci.vehicle.getLaneID(self.Id) == '':
					traci.simulationStep()
				self.present_lane = traci.vehicle.getLaneID(self.Id)
			self.present_sublane = int(self.present_lane.split("_")[1])
			self.max_speed = traci.vehicle.getAllowedSpeed(self.Id)
			self.speed = traci.vehicle.getSpeed(self.Id)
			self.lateral_speed = traci.vehicle.getLateralSpeed(self.Id)
			self.acceleration = traci.vehicle.getAcceleration(self.Id)
			self.acceleration_records.append(self.acceleration)
			self.angle = traci.vehicle.getAngle(self.Id)
		except:
			#traci.simulationStep()
			self.reset()
			print('reset called')

	
	def compute_observation_matrix(self, threshold_distance=10):

		agent_lane = self.present_lane
		agent_pos = self.pos
		edge = self.present_lane.split("_")[0]
		agent_lane_index = self.present_sublane
		lanes = [lane for lane in self.lane_id_list if edge in lane]
		state = np.zeros([self.obs_dim, self.obs_dim])
		
		agent_x, agent_y = 1, agent_lane_index
		state[agent_x, agent_y] = -1
	
		for lane in lanes:
		
			vehicles = traci.lane.getLastStepVehicleIDs(lane)
			veh_lane = int(lane.split("_")[-1])
			for vehicle in vehicles:
				if vehicle == self.Id:
					continue
				
				veh_pos = traci.vehicle.getPosition(vehicle)
				
				if get_distance(agent_pos, veh_pos) > threshold_distance:
					continue
				rl_angle = traci.vehicle.getAngle(self.Id)
				veh_id = vehicle.split("_")[1]
				angle = angle_between(agent_pos, veh_pos, rl_angle)
				# Putting on the right
				if angle > 337.5 or angle < 22.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the right north
				if angle >= 22.5 and angle < 67.5:
					state[agent_x-1,veh_lane] = veh_id
				# Putting on north
				if angle >= 67.5 and angle < 112.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left north
				if angle >= 112.5 and angle < 157.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left
				if angle >= 157.5 and angle < 202.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the left south
				if angle >= 202.5 and angle < 237.5:
					state[agent_x+1, veh_lane] = veh_id
				if angle >= 237.5 and angle < 292.5:
					# Putting on the south
					state[agent_x+1, veh_lane] = veh_id
				# Putting on the right south
				if angle >= 292.5 and angle < 337.5:
					state[agent_x+1, veh_lane] = veh_id
		
		print('Before flipping',state)
		state = np.fliplr(state)
		print(state)
		return state
		
	def get_comfort(self):
		return (self.acceleration_records[1] - self.acceleration_records[0])/self.step_length

	def get_collision_value(self):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		if self.Id in collisions:
			self.collision = True
			return True
		self.collision = False
		return False
	
	def get_compute_state(self):

		state = np.zeros(self.state_dim)
		before = 0
		grid_state = self.compute_observation_matrix().flatten()
		for num, vehicle in enumerate(grid_state):
			if vehicle == 0:
				continue
			if vehicle == -1:
				vehicle_Id = self.Id
				before = 1
			else:
				vehicle_Id = 'vehicle_'+(str(int(vehicle)))
			veh_info = self.get_info(vehicle_Id)
			idx_init = num*4
			if before and vehicle != -1:
				idx_init += 1
			idx_fin = idx_init + veh_info.shape[0]
			state[idx_init:idx_fin] = veh_info
		state = np.squeeze(state)
		return state
	
	
	def get_info(self, vehicle_Id):

		if vehicle_Id == self.Id:
			return np.array([self.pos[0], self.pos[1], self.speed, self.lateral_speed, self.acceleration])
		else:
			lat_pos, long_pos = traci.vehicle.getPosition(vehicle_Id)
			long_speed = traci.vehicle.getSpeed(vehicle_Id)
			acc = traci.vehicle.getAcceleration(vehicle_Id)
			dist = get_distance(self.pos, (lat_pos, long_pos))
			return np.array([dist, long_speed, acc, lat_pos])
		
		
	def compute_reward(self, collision, action):


		alpha_comf = 0.0005
		w_lane = 1.5
		w_speed = 1.5
		w_change = 1.0
		w_eff = 0.0005

		jerk = self.get_comfort()
		R_comf = -alpha_comf*jerk**2

		try:
			lane_width = traci.lane.getWidth(traci.vehicle.getLaneID(self.Id))
		except:

			lane_width = 3.2
		desired_x = self.pos[0] + lane_width*np.cos(self.angle)
		desired_y = self.pos[1] + lane_width*np.sin(self.angle)
		R_lane = -(np.abs(self.pos[0] - desired_x) + np.abs(self.pos[1] - desired_y))

		R_speed = -np.abs(self.speed - self.max_speed)

		if action:
			R_change = -1
		else:
			R_change = 1

		R_eff = w_eff*(w_lane*R_lane + w_speed*R_speed + w_change*R_change)
		

		if collision:
			R_safe = -100
		else:
			R_safe = +1
		

		R_tot = R_comf + R_eff + R_safe
		return [R_tot, R_comf, R_eff, R_safe]
		
		
	def new_step(self, action):

		# Actions : 0 stay, 1 change to right, 2 change to left
		if self.present_lane[0] == 'e':
			action = 0
		if action != 0:
			if action == 1:
				if self.present_sublane == 1:
					traci.vehicle.changeLane(self.Id, 0, 0.1)
				elif self.present_sublane == 2:
					traci.vehicle.changeLane(self.Id, 1, 0.1)
			if action == 2:
				if self.present_sublane == 0:
					traci.vehicle.changeLane(self.Id, 1, 0.1)
				elif self.present_sublane == 1:
					traci.vehicle.changeLane(self.Id, 2, 0.1)

		traci.simulationStep()

		collision = self.get_collision_value()
 
		reward = self.compute_reward(collision, action)

		self.update_params()

		next_state = self.get_compute_state()

		self.present_step += 1
		done = collision
		return next_state, reward, done, collision
		
	def render(self, mode='vehicle', close=False):
		pass

	def reset(self, gui=False, no_Vehicle=35, veh_type='vehicle'):
		self.start(gui, no_Vehicle, veh_type)
		return self.get_compute_state()

	def close(self):
		traci.close(False)
