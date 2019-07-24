import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import random
import sys
import os

class UDNEnv(gym.Env):
    metadata = {}
    
    def __init__(self):
        self.BSposition = np.loadtxt('BSposition.csv', delimiter=',')
        self.BSnum = len(self.BSposition[0])
        self.InterferenceBSposition = np.loadtxt('InterferenceBSposition.csv', delimiter=',')
        self.InterferenceBSnum = len(self.InterferenceBSposition[0])
        self.Area = 10**3
        self.usernum = 16
        self.BSstate = np.ones(self.BSnum)
        self.InterferenceBSstate = np.random.randint(2,size=self.InterferenceBSnum)
        self.user_Xposition = np.random.uniform(0,self.Area,self.usernum)
        self.user_Yposition = np.random.uniform(0,self.Area,self.usernum)
        self.action_space = spaces.Discrete(2**self.BSnum)
        self.movedistance = None
        self.state = np.r_[self.BSstate,self.user_Xposition,self.user_Yposition]
        self.totalreward = 0
        self.bandwidth = 10**7
        
    
    def step(self, action):
        #
        #return state action pair reward 
        self.take_action(action)
        
        Datarate_weightvalue = 10**6
        Energyconsumption_weightvalue = 1
        
        signal = self.BS_User_S()
        Interference = self.Interference_User_I()
        SIR = signal/Interference
        Datarate = self.bandwidth * np.log2(1+SIR)
        Energyconsumption = np.sum(self.state[:self.BSnum])
        if Energyconsumption == 0:
            reward = -10
        else:
            reward = Datarate_weightvalue * np.mean(Datarate) / (Energyconsumption_weightvalue * Energyconsumption)
        if reward < 0.1:
            is_done = True
        else:
            is_done = False
        info = ""
        return self.state, reward, is_done, info#for visualizing
    
    def reset(self):
        self.BSstate = np.ones(self.BSnum)
        self.user_Xposition = np.random.uniform(0,self.Area,self.usernum)
        self.user_Yposition = np.random.uniform(0,self.Area,self.usernum)
        self.state = np.r_[self.BSstate,self.user_Xposition,self.user_Yposition]
        
        return self.state
    
    def take_action(self, action):
        #do action for change state
        self.BSstate = self.Binarychange(action)
        self.movedistance = self.usermovedistance()
        
        for i in range(self.BSnum):
            self.state[i] = self.BSstate[i]
        
        for j in range(self.BSnum,self.BSnum+2*self.usernum):
            self.state[j] = self.state[j] + self.movedistance[j-self.BSnum]
            if self.state[j] > self.Area:
                self.state[j] = self.state[j] - self.Area
            
    
    def Binarychange(self,num):
        #hex number to binary matrix
        hnum = num
        bmatrix = np.zeros(self.BSnum)
        index = 0
        while True:
            if index == self.BSnum:
                break
            else:
                bmatrix[index] = hnum % 2
                hnum = hnum // 2
                index += 1
        return bmatrix
    
    def usermovedistance(self):
        #human walking speed 1.3m/s = 4.68km/h
        theta = np.random.uniform(0,2*np.pi,self.usernum) #random angle for each user
        d = np.random.uniform(0,1.3,self.usernum)#random distance for each user
        sin = np.sin(theta)
        cos = np.cos(theta)
        x_dis = d*cos
        y_dis = d*sin
        state_dis = np.r_[x_dis,y_dis] #form for state
        return state_dis
    
    def BS_User_S(self):
        #calculate Signal power consist path loss for each user
        #return 1 by usernum matrix include signal power for each user
        BS_User_position = np.zeros((2,self.usernum,self.BSnum))
        BS_User_distance = np.zeros((self.usernum,self.BSnum),dtype = float)
        user_signalpower = np.zeros(self.usernum,dtype = float)
        # axis x = 0, axis y = 1
        for i in range(self.usernum):
            for j in range(self.BSnum):
                BS_User_position[0][i][j] = self.state[self.BSnum + i] - self.BSposition[0][j]
                BS_User_position[1][i][j] = self.state[self.BSnum + self.usernum + i] - self.BSposition[1][j]
        BS_User_distance = np.linalg.norm(BS_User_position, ord = 2, axis = 0)
        assosiation_matrix = self.assosiation(BS_User_distance)
        user_signalpower = np.power(BS_User_distance[assosiation_matrix],-4)
        return user_signalpower
    
    def Interference_User_I(self):
        #calculate Interference power consist path loss for each user
        #return 1 by usernum matrix include interference power for each user
        InterferenceBS_User_position = np.zeros((2,self.usernum,self.InterferenceBSnum))
        InterferenceBS_User_distance = np.zeros((self.usernum,self.BSnum), dtype = float)
        InterferenceBSstate_bool = self.InterferenceBSstate.astype(bool)
        user_interfenecepower = np.zeros(self.usernum,dtype = float)
        #axis x = 0, axis y = 1
        for i in range(self.usernum):
            for j in range(self.InterferenceBSnum):
                InterferenceBS_User_position[0][i][j] = self.state[self.BSnum + i] - self.InterferenceBSposition[0][j]
                InterferenceBS_User_position[1][i][j] = self.state[self.BSnum + self.usernum + i] - self.InterferenceBSposition[1][j]
        Interference_User_distance = np.linalg.norm(InterferenceBS_User_position, ord = 2, axis = 0)
        if np.sum(self.InterferenceBSstate) == 0:
            user_interferencepower = np.sum(Interference_User_distance,axis = 1)
        else:
            user_interferencepower = np.sum(Interference_User_distance[:,InterferenceBSstate_bool],axis =1)
        print(user_interferencepower)
        return user_interferencepower
    

    def assosiation(self, distance):
        #calculate user-BS assosiation follow shortest distance assosiation rule
        #return usernum by BSnum matrix dtype boolean
        BS_user_assosiation = np.zeros((self.usernum,self.BSnum),dtype = bool)
        user_BS_shortest = np.argmin(distance,axis = 1)
        for i in range(self.usernum):
            BS_user_assosiation[i][user_BS_shortest[i]] = True

        return BS_user_assosiation
