import math
import gym
#from gym import space, logger
#from gym.utils import seeding
import numpy as np
import tensorflow as tf
#import pandas as pd


class UDNEnv(gym.Env):

	def __init__(self):
		#변수 선언
		self.d_attenuation = -4
		self.state = None
		self.BSposition = np.transpose(np.loadtxt('BSposition.csv', delimiter=','))
		self.BSnum = len(BSposition)
		self.Area = 10000
		self.usernum = 16
		self.done = False
		self.BSpower = tf.ones([1,BSnum])
		self.d_at_tensor = tf.fill([1,usernum],-4)
		self.US_Xposition = tf.random_uniform([1,usernum],0,Area)
		self.US_Yposition = tf.random_uniform([1,usernum],0,Area)
		self.reward = None
		self.BS_user_distance = tf.zeros(BSnum,usernum,2)
		self.BSdistance = tf.zeros(BSnum,usernum)
		self.user_assosiation = tf.zeros([1,usernum])
		self.SNR = None

	def step(self, action):
		state = self.state
		state = action
		Econsumption = tf.reduce_sum(state)
		distance = np.array(BSdistance)
		
		#BS-User 거리 계산
		for i in range(BSnum):
			for j in range(usernum):
				BS_user_distance[i][j][0] = BSposition[i][0] - US_Xposition[j]
				BS_user_distance[i][j][1] = BSposition[i][1] - US_Yposition[j]
		BSdistance = np.linalg.norm(BS_user_distance, axis=2, ord = 2)
		#거리 계산 종료
		
		user_assosiation = assosiation(state, BSdistance)
		SNR = tf.pow(user_assosiation,d_at_tensor) #SNR 계산

		reward = tf.reduce_sum(SNR) / tf.reduce_sum(state) #reward 계산

		self.US_Xposition = tf.random_uniform([1,usernum],0,Area) #유저 위치 랜덤 배치
		self.US_Yposition = tf.random_uniform([1,usernum],0,Area)
		return reward, state

		#BS-User assosiation 하는 함수
	def assosiation(self, state, distance):
		assosiation_distance = tf.zeros([1,usernum])
		assosiationdistance = [BSnum, usernum]

		#전원 off되어 있는 BS에서 유저까지의 거리를 무한대로 설정
		for i in len(state):
			if state[i] == 0:
				for j in len(usernum):
					assosiationdistance[i][j] = math.inf
		
		assosiation_distance = tf.math.reduce_min(assosiationdistance, axis = 0)
		return assosiation_distance


	def reset(self):
		self.state = self.tf.ones([1,BSnum])
		self.done = False
		return state

