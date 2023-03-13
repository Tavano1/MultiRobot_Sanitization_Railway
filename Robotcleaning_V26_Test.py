
#Setup


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1.keras.backend import set_session
#import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
from numpy import random
from scipy import signal
import pandas as pd
from scipy import misc

import matplotlib.pyplot as plt
import matplotlib.colors

from datetime import date

from scipy.ndimage.filters import gaussian_filter
import copy

import signal
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Lock
from multiprocessing import Pipe
import itertools

#-- FOR PRECISION-server ONLY!!
#from tensorflow.config.experimental import list_physical_devices, set_memory_growth
#physical_devices = list_physical_devices('GPU')
#set_memory_growth(physical_devices[0], True)
#--

##########################################################################
##########################################################################
STEPS_PER_EPISODE = 960#150 #100
CLEAN_RAY = 3
CLUSTER_RAY = 1
N_robots=0 #4
TESTING = True
#N_total =130
N_totalX =172
N_totalY=100
N_Walkman=500 #1500
update_period = 15
Lowlim_N_Walk=int(N_Walkman-N_Walkman*30/100)
#N_upp=0
#catch Ctrl+\ to change the visualization [RIC]
def changeVisualization(signalNumber, frame):
    global TESTING

    if TESTING:
        TESTING = False
    else:
        TESTING = True

    print('(SIGQUIT) MAP-PLOTTING CHANGED TO ', TESTING)
    return

def doNothing(signalNumber, frame):
    return

class BlobEnv:
    def __init__(self):
        self.Data=np.zeros((N_totalX,N_totalY,2))
        self.Wall_Matrix=np.zeros((N_totalX,N_totalY))

        
            

    def reset(self):
        #print("reset BlobEnv")
#### here I need to put to zero, heatmap, robot positions

        self.Walls_and_Pillars_Matrix()  
#### initialize the data matrix with dummy values only for testing
        self.Data=np.zeros((N_totalX,N_totalY,2))

        self.episode_step = 0

        self.state_first=copy.deepcopy(self.Data)
        self.first_sum = np.sum(self.state_first[:, :, 0]) 
    ### define and initialize the position of the robot inside of my space 100x100
        #self.DQNAgents = []
        #for Rango in range(self.countRobot):
        #    self.DQNAgents.append(DQNAgent())
        #self.Create_Walkman(Walkman,N_Walkman_rand)
        #self.applyWalkman()
        self.updateHeatmap(0)

    def Walls_and_Pillars_Matrix(self):

        #N_total=250
        self.Wall_Matrix=np.zeros((N_totalX,N_totalY))
        originalImage = cv2.imread("Termini BeW_1px_1m_100x172.png")
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        myData=np.array(blackAndWhiteImage)
        Convert=np.where(myData==0, 1, myData)
        self.Wall_Matrix=np.where(myData==255,0, Convert) 

    def Clean_Walls_and_Pillars(self):
        for i in range(0, N_totalX):
            for j in range(0, N_totalY):
                #print("wall: ", self.Wall_Matrix[i][j])
                if self.Wall_Matrix[i][j] > 0.5:
                    #print("put to zero: ", i, " ", j)
                    self.Data[i, j, 0] = 0



    def Put2Zero(self):
        super_threshold_indices = self.Data[:,:,0] < 1e-3
        self.Data[super_threshold_indices,0] = 0

    def updateHeatmap(self,count):
        #print("updateHeatmap", self.episode_step)
        #if timestep%5==0:
        #for robs in range(self.N_Walkman):
              
        #    self.PositionWalkmanMatrix(robs)
        Dir='Nuovi_Test_Heatmap3/Test_all_Day/MyHeatmap'+str(count)+'.csv'
        #print("count",count)
        #NewUpgrade=cv2.imread("Testheatmap2/MyHeatmap"+str(count)+".csv") 
        NewUpgrade=pd.read_csv(Dir,header=None)
        #print("nonetype",type(NewUpgrade),np.shape(NewUpgrade)) 
        self.Data[:, :, 0]=self.Data[:,:,0]+np.array(NewUpgrade)

        super_threshold_indices = self.Data[:,:,0] > 1
        self.Data[super_threshold_indices,0] = 1

        super_threshold_indices2 = self.Data[:,:,0] < 1e-3
        self.Data[super_threshold_indices2,0] = 0
          
            #compensate gaussia blur with multiplication [RIC]
        self.Data[:, :, 0] = gaussian_filter(self.Data[:, :, 0], sigma=0.9, truncate=4.0)



        # creo una matrice della posizione attuale del robot. la aggiungo alla heatmap,
        #in seconda posizione della terza dimensione del cubo.
        self.Put2Zero()

        #print("state_next ", self.episode_step)
        self.Clean_Walls_and_Pillars()

        self.state_next=self.Data
        
        #return self.state_next
        #out_queue.put([self.state_next,self.reward,self.done])
    
    def updateHeatmap2(self,count):
        #print("updateHeatmap", self.episode_step)
        #if timestep%5==0:
        #for robs in range(self.N_Walkman):

          
            #compensate gaussia blur with multiplication [RIC]
        self.Data[:, :, 0] = gaussian_filter(self.Data[:, :, 0], sigma=0.9, truncate=4.0)



        # creo una matrice della posizione attuale del robot. la aggiungo alla heatmap,
        #in seconda posizione della terza dimensione del cubo.
        self.Put2Zero()

        #print("state_next ", self.episode_step)
        self.Clean_Walls_and_Pillars()

        self.state_next=self.Data
        
        #return self.state_next
        #out_queue.put([self.state_next,self.reward,self.done])
    def update_positionOfRobot_Matrix(self,px,py,robs):
           # self.RobMatrix=np.zeros((100,100))

        if robs==0:
            self.Data[:,:,1]=np.zeros((N_totalX,N_totalY)) #da fare fuori, nel main
            #print("sono nell'if")
        #print("PositionAgents[robs+1] + j ",PositionAgents[robs+1] + j )
            
        #RIC
        for i in range(-CLEAN_RAY, +CLEAN_RAY):
            if px + i >= N_totalX or px + i < 0:
                continue

            for j in range(-CLEAN_RAY, +CLEAN_RAY):
                if py + j >= N_totalY or py + j < 0:
                    continue
                #print("PositionAgents[robs] + i ",PositionAgents[robs] + i )
                #print("PositionAgents[robs+1] + j ",PositionAgents[robs+1] + j )
                self.Data[px + i, py + j,1] = 1

           # env.Data[:,:,1]=self.RobMatrix
            #self.RobotClean()
            self.clean_robot_position(px,py,robs) #RIC

    def clean_robot_position(self,px,py,robs):
        cleaning_ray=CLEAN_RAY #the diameter will than be cleaning_ray*2 + 1
        for i in range(-CLEAN_RAY, +CLEAN_RAY):
            if px + i >= N_totalX or px + i < 0:
                continue

            for j in range(-CLEAN_RAY, +CLEAN_RAY):
                if py + j >= N_totalY or py + j < 0:
                    continue

                self.Data[px + i, py + j,0] = 0

###########################################################################################
class DQNAgent:

    def __init__(self,seed):
        self.positionRobX = []
        self.positionRobY = []

        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0
        self.episode_reward = 0
        self.N_robots=N_robots
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.seed = seed # da mettere nel init e diventa self.
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter, lo inizializza nel init, e diventa self.
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 50000
        # Number of frames for exploration
        self.epsilon_greedy_frames = 1000000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        #self.max_memory_length = 5000 #30000/N_robots #100000
        #self.max_memory_length = 100000 / N_robots  # Precision - Server
        self.max_memory_length = 5000 #memory buffer size
        # Train the model after 4 actions
        self.update_after_actions = 4
        # How often to update the target network
        self.update_target_network = 10000
        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()
        # building of a list of agents with a number of elements equal to countRobot
        #countRobot=N_robots     
        self.batch_size = 32 
        self.env=BlobEnv()
        self.DoneRunning_Flag=False
        self.epsilon_interval = (
        self.epsilon_max - self.epsilon_min
    )  # Rate at which to reduce chance of random action being taken


    def my_Agent_action_update(self,NewpositionX,NewpositionY):
            # Update Q value for given state

            # And append to our training data
            self.positionRobX.append(NewpositionX)
            self.positionRobY.append(NewpositionY)

    def my_Agent_action_updateX(self,NewpositionX):
        # Update Q value for given state

        # And append to our training data
        self.positionRobX.append(NewpositionX)
        
    def my_Agent_action_updateY(self,NewpositionY):
            # Update Q value for given state

            # And append to our training data
            
            self.positionRobY.append(NewpositionY)
    def Walls_and_Pillars_Matrix(self):

        #N_total=250
        self.Wall_Matrix=np.zeros((N_totalX,N_totalY))
        originalImage = cv2.imread("Termini BeW_1px_1m_100x172.png")
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        myData=np.array(blackAndWhiteImage)
        Convert=np.where(myData==0, 1, myData)
        self.Wall_Matrix=np.where(myData==255,0, Convert)
        self.env.Wall_Matrix=self.Wall_Matrix

   
    def actionRobot(self,choice): #da mettere nel agent

        rob_speed = CLEAN_RAY #was 5
        #rob_speed = 2 #was 5
        #Gives us 9 total movement options. (0,1,2,3,4,5,6,7)
        if choice == 0:
                self.dx=1
                self.dy=1
        elif choice == 1:
                self.dx=-1
                self.dy=-1
        elif choice == 2:
                self.dx=-1
                self.dy=1
        elif choice == 3:
                self.dx=1
                self.dy=-1

        elif choice == 4:
                self.dx=1

                self.dy=0
        elif choice == 5:
                self.dx=-1
                self.dy=0

        elif choice == 6:
                self.dx=0
                self.dy=1

        elif choice == 7:
                self.dx=0
                self.dy=-1
        Wall_flagX=False
        Wall_flagY=False

        #check the X borders
        if self.positionRobX[-1]+rob_speed*self.dx<=(N_totalX-1) and self.positionRobX[-1]+rob_speed*self.dx>=0:
            Wall_flagX=True

        #check the Y borders
        if self.positionRobY[-1]+rob_speed*self.dy<=(N_totalY-1) and self.positionRobY[-1]+rob_speed*self.dy>=0:
            Wall_flagY=True

        #if borders are consistent
        if Wall_flagX==True and Wall_flagY==True:
            #check the obstacles
            if self.Wall_Matrix[self.positionRobX[-1]+rob_speed*self.dx][self.positionRobY[-1]+rob_speed*self.dy]==1:
                self.my_Agent_action_updateX(self.positionRobX[-1])
                self.my_Agent_action_updateY(self.positionRobY[-1])
            else:
                #update the positions
                self.my_Agent_action_updateX(self.positionRobX[-1] + rob_speed * self.dx)
                self.my_Agent_action_updateY(self.positionRobY[-1] + rob_speed * self.dy)
        else:
            self.my_Agent_action_updateX(self.positionRobX[-1])
            self.my_Agent_action_updateY(self.positionRobY[-1])

        self.PositionRobotMatrix()

    def PositionRobotMatrix(self):
        #print("self.env.Data",type(self.env.Data),np.shape(self.env.Data))
        #print("self.env.Data",self.env.Data[:,:,1])
        self.RobMatrix=np.zeros((N_totalX,N_totalY))
        self.env.Data[:,:,1]=np.zeros((N_totalX,N_totalY))
        #RIC
        for i in range(-CLEAN_RAY, +CLEAN_RAY):
            if self.positionRobX[-1] + i >= N_totalX or self.positionRobX[-1] + i < 0:
                continue

            for j in range(-CLEAN_RAY, +CLEAN_RAY):
                if self.positionRobY[-1] + j >= N_totalY or self.positionRobY[-1] + j < 0:
                    continue

                self.RobMatrix[self.positionRobX[-1] + i, self.positionRobY[-1] + j] = 1
        #print("self.env.Data",type(self.env.Data),np.shape(self.env.Data))
        #print("self.env.Data",self.env.Data[:,:,1])
        self.env.Data[:,:,1]=self.RobMatrix
        self.clean_robot_position_onMy_Env() #RIC

    def clean_robot_position_onMy_Env(self):
        cleaning_ray=CLEAN_RAY #the diameter will than be cleaning_ray*2 + 1
        for i in range(-cleaning_ray, +cleaning_ray):
            if self.positionRobX[-1] + i >= N_totalX or self.positionRobX[-1] + i < 0:
                continue

            for j in range(-cleaning_ray, +cleaning_ray):

                if self.positionRobY[-1] + j >= N_totalY or self.positionRobY[-1] + j < 0:
                    continue
                #commentato per test
                self.env.Data[self.positionRobX[-1] + i, self.positionRobY[-1] + j, 0] = 0

    
    def MyReward(self):
        self.reward = self.old_sum - self.new_sum

        #punish no-cleaning actions
        #self.reward=( 1 - (self.episode_step/60) )*self.reward
        if self.reward < 0.001:
            #self.reward = -1
            self.reward = -2.0 #lets try a lower punishment
        #else:
         #  self.reward=( 1 - (self.episode_step/600) )*self.reward

        #print("REWARD: ", self.reward,(STEPS_PER_EPISODE-self.episode_step) /STEPS_PER_EPISODE,self.episode_step )
        #input("press enter...")
    def step(self, action):

    ################# here we my calculate the state of the sistem after the action: resulting heatmap, gain
    #state_next, reward, done, 

        #the last robot increase steps! so now there is one step increment every N robots executions

        self.episode_step += 1
        #self.env.Data[:, :, 0] = gaussian_filter(self.env.Data[:, :, 0], sigma=0.9, truncate=4.0)

        if self.episode_step ==1: 
            #self.state_old=copy.deepcopy(self.Data)
            #self.beforeCleaning_position=self.state_old[self.positionRobX,self.positionRobY,0]
            self.state_first=copy.deepcopy(self.env.Data)




        self.old_sum = np.sum(self.env.Data[:, :, 0])  # [RIC]

        self.actionRobot(action)

        self.new_sum = np.sum(self.env.Data[:, :, 0])  # [RIC]

        #self.Put2Zero()
        self.state_next=self.env.Data
        #print("state_next ", self.episode_step)
        #self.render2()


        self.MyReward() #[RIC]


        #self.state_old=copy.deepcopy(self.Data)
        #self.beforeCleaning_position=self.Step_Next
        #self.beforeCleaning_position=self.state_old[self.positionRobX,self.positionRobY,0]
        self.done = False
        #input("press enter 5")
        #if self.mgained<=self.mgained_first/3:
        non_zero_percentage = ( float(np.count_nonzero(self.env.Data[:, :, 0])) / float(N_totalX * N_totalY) ) * 100.0
        #if self.new_sum <= 0.0001:
        if non_zero_percentage < 2.0: #run is solved if 95% of map is clear!
        
            self.done = True
            self.reward = 100
            print("MAP IS CLEAR at step ", self.episode_step)
        
        return self.state_next,self.reward,self.done
        #out_queue.put([self.state_next,self.reward,self.done])
        
    def create_q_model(self,num_actions ):
    # Network defined by the Deepmind paper
        inputs = layers.Input(shape=(N_totalX, N_totalY, 2,)) # terza dimensione pari a 2

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(num_actions, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=action)
        
    def make_my_Model(self,num_actions):
        self.num_actions=num_actions
        # The first model makes the predictions for Q-values which are used to
        # make a action.
        self.model = self.create_q_model(num_actions)
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.model_target = self.create_q_model(num_actions)
        # In the Deepmind paper they use RMSProp however then Adam optimizer
        # improves training time        
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        
    #def doReset(self,DQNAgents,N_robots,robs):
    def doReset(self):
        #self.env=env
        #for robs in range(self.countRobot):
        self.episode_step = 0
        randx=np.random.randint(0, N_totalX-1)
        randy=np.random.randint(0, N_totalY-1)
        while self.env.Wall_Matrix[randx][randy]==1:
            randx=np.random.randint(0, N_totalX-1)
            randy=np.random.randint(0, N_totalY-1)
        #print("sono nel reset ",robs, randx,randy)
        self.my_Agent_action_update(randx,randy)
        
        #self.PositionRobotMatrix(env)

        self.rewards_log = np.zeros((N_totalY,2))
        self.steps = 0
        self.reward = 0
        self.cumulative_reward = 0
        

    #def DoMission(self): # Moved to the main_process() [RIC]


    def Calculate_running_reward(self):

        # Update running reward to check condition for solving
        self.episode_reward_history.append(self.episode_reward)
        if len(self.episode_reward_history) > 100:
            del self.episode_reward_history[:1]
        self.running_reward = np.mean(self.episode_reward_history)

    def Check_condition_Solving(self):
        if self.running_reward > 100:  # Condition to consider the task solved
            #print("Solved at episode {}!".format(self.episode_count))
            #print("     reward: {}!".format(self.running_reward))
            self.DoneRunning_Flag=True


    def main_Process(self, num_actions, my_conn, max_steps_per_episode, id_rob):

        #cumulative_step = 0
     #   single_robot_rewards = np.zeros((4,1))
        episode_count = 0
        frame_count = 0
        #stored_execution_time = time.time()
        running_reward=0
        episode_reward=0
        DoneRunning=False
        
        episode_reward_history = []
        
        num_actions = 8
        self.model = tf.keras.models.load_model('saved_model/my_model'+str(id_rob))
        self.model_target = tf.keras.models.load_model('saved_model/my_model_target'+str(id_rob))
           
        #self.make_my_Model(num_actions)

        signal.signal(signal.SIGQUIT, doNothing)
 
        while True:  # Run until solved
            self.done=False
            self.Walls_and_Pillars_Matrix()
            #my_conn.close()
            #print("aggiorno le liste di posizione PositionRobotX PositionRobotY")
            # trovo random le posizioni iniziali dei robot, 
            #e aggiorno le liste di posizione PositionRobotX PositionRobotY
            self.doReset()

            my_conn.send([self.positionRobX[-1],self.positionRobY[-1]]) # prima comunicazione
            #print("process-", id_rob, " sent init pos")

            #self.update_positionOfRobot_Matrix(env,robs)
            
            self.episode_reward = 0

            # ricevo l'env aggiornato
            #print("process-", id_rob, " receiving init state")
            myrecData0=my_conn.recv() #seconda comunicazione
            #print("process-", id_rob, " received init state")

            #print("seconda comunicazione myrecData0",type(myrecData0),np.shape(myrecData0))
            myrecData0=np.array(myrecData0)
            #print("seconda comunicazione np.array(myrecData0)",type(myrecData0),np.shape(myrecData0))
            self.env.Data=np.array(myrecData0[:,:,:])
            #print("self.env.Data=copy.deepcopy(my_conn.recv()) ",type(self.env.Data),np.shape(self.env.Data))
            #print("self.env.Data",self.env.Data[:,:,1])


            # aggiorno la posizione dell'agente in self.Data[:,:,1] ed effettuo la pulizia su self.Data[:,:,0]
            # quindi sulla mia copia
            self.PositionRobotMatrix()

            #aggiorno lo stato con la copia
            self.state = np.array(self.env.Data)
            #input("press enter 1")

            #stored_execution_time = time.time()

            for timestep in range(1, max_steps_per_episode+1):

                #print("step ", timestep, " executed in ", time.time() - stored_execution_time)

                #print("terza comunicazione")
                #ricevo l'env aggiornato dopo l'applicazione della gaussiana
                myrecData0=my_conn.recv()  # terza comunicazione
                #print("process-", id_rob, " received state")

                # faccio una copia dell'env su cui il robot può lavorare autonomamente

                #stored_execution_time = time.time()

                myrecData0=np.array(myrecData0)
                #print("terza comunicazione np.array(myrecData0)",type(myrecData0),np.shape(myrecData0))

            
                self.env.Data=np.array(myrecData0[:,:,:])

                if self.env.Data[:,:,0].max() > 0.0001:
                    self.env.Data[:,:,0] *= (1.0/self.env.Data[:,:,0].max()) #[RIC] rescale the matrix so the max value will be 1.0!

                # aggiorno la posizione dell'agente in self.Data[:,:,1] ed effettuo la pulizia su self.Data[:,:,0]
                # quindi sulla mia copia
                self.PositionRobotMatrix()              
            
                frame_count += 1
                self.frame_count = frame_count
     
                overall_state = np.zeros((N_totalX,N_totalY,2))

                # chiamo il metodo in cui si effettua lo spostamento dei robot,

                #la pulizia e l'aggiornamento della Q-table
                #self.DoMission() ###########################################################################

                self.done = False
                self.DoneRunning_Flag = False
                done = False

                self.state = np.array(self.env.Data)
                '''
                # Use epsilon-greedy for exploration
                if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    # Take random action
                    self.action = np.random.choice(self.num_actions)
                else:
                '''
                    # Predict action Q-values
                    # From environment state
                state_tensor = tf.convert_to_tensor(self.state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                    # Take best action
                self.action = tf.argmax(action_probs[0]).numpy()
                    #  print("action= ConvNet choice")
                # Decay probability of taking random action
                #self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                #self.epsilon = max(self.epsilon, self.epsilon_min)

                state_next, reward, done = self.step(self.action)  ##########################################

                self.reward = reward
                self.done = done
                self.state_next = np.array(state_next)

                my_conn.send(self.done)  # quarta comunicazione
                #print("process-", id_rob, " done sent: ",self.done)

                my_Done_Flag = my_conn.recv()  # quinta comunicazione
                #print("process-", id_rob, " received done_flag: ", my_Done_Flag)


                self.done = my_Done_Flag

                #if my_Done_Flag:
                    #self.reward = 100 #get_winning_reward()
                   # self.reward = 50 + ( (STEPS_PER_EPISODE - timestep) / STEPS_PER_EPISODE ) * 100
                #    self.reward = ( (STEPS_PER_EPISODE - timestep) / STEPS_PER_EPISODE ) * 1000



                self.episode_reward += self.reward

                # Save the single robot reward into the log (for plotting) [RIC]
                # robot_rewards_log[timestep-1, robs+1] = reward
                '''
                # Save actions and states in replay buffer
                self.action_history.append(self.action)
                self.state_history.append(self.state)
                self.state_next_history.append(self.state_next)
                self.done_history.append(self.done)
                '''
                self.rewards_history.append(self.reward)
                #self.state = self.state_next
                
                # Limit the state and reward history
                if len(self.rewards_history) > self.max_memory_length:
                    '''
                    
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]
                    '''
                    del self.rewards_history[:1]
                    del self.positionRobY[:1]
                    del self.positionRobX[:1]
                # Update every fourth frame and once batch size is over 32
                


                #############################################################################################





                ### ciascun robot invia la propria posizione per aggiornamento dell'environment a partire dalla conoscenza della dei robots
                #input("press enter 1")
                #print("quarta comunicazione",type(self.positionRobX),type(self.positionRobY))
                my_conn.send([self.positionRobX[-1],self.positionRobY[-1]]) #sesta comunicazione
                #print("process-", id_rob, " position sent")
                
                #StringOK=my_conn.recv()
                #print("process-", id_rob, " ack received")

                #input("press enter 1")
                self.Calculate_running_reward()

                #print("quinta comunicazione",type(self.done),type(self.reward),type(self.episode_reward))
                my_conn.send([self.done,self.reward,self.episode_reward]) #settima comunicazione
                #print("process-", id_rob, " output sent")

                '''
                if done:
                    print("id-",id_rob, ", done by ME! reward= ", self.reward)

                    self.done=False
                    #break
                elif( my_Done_Flag ):
                    print("id-", id_rob, ", done by ANOTHER! reward= ", self.reward)

                    self.done = False
                    #break
                '''

            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 1000:
                del episode_reward_history[:1]

            self.running_reward = np.mean(episode_reward_history)
            
            my_task_solved=False
            
            my_task_solved_list= my_conn.recv() #comunicazione 8
            my_task_solved=my_task_solved_list[1] 
            my_model_name=my_task_solved_list[0]
            
            '''
            if( my_task_solved):
                my_task_solved=False
                self.model.save('saved_model/my_model'+str(my_model_name))
                self.model_target.save('saved_model/my_model_target'+str(my_model_name))
                break   	
            
            if self.running_reward > 100:  # Condition to consider the task solved
                print("Solved at episode {}!".format(self.episode_count))
                print("     reward: {}!".format(self.running_reward))
                break
            '''

            episode_count += 1
            self.episode_count = episode_count

###########################################################################################




def main():
    mp.set_start_method('spawn')
    my_Done_Flag=False

    Children=[]
    Parent=[]
    for Rango in range(N_robots):
        Parent_conn,Children_conn=Pipe()
        Parent.append(Parent_conn)
        Children.append(Children_conn)


    seed = 42
    cumulative_rewards_log = np.zeros((10000000, 2))
    single_robot_rewards_log = np.zeros((10000000, N_robots))
    cumulative_step = 0
    max_steps_per_episode = STEPS_PER_EPISODE
 #   single_robot_rewards = np.zeros((4,1))
    episode_count = 0
    frame_count = 0
    stored_execution_time = time.time()
    running_reward=0
    episode_reward=0
    DoneRunning=False
    
    #episode_not_zero_percentage_history = [] # only to evaluate test;
    episode_PercentageSomma_history = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_max = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_min = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_mean = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_Var=[] # new!
    episode_reward_history = []

    ######################################NeW ########################
    #primo blocco, tutto ok
    TENrunning_rewardMIN=[]
    TENrunning_rewardMean=[]
    TENrunning_rewardMAX=[]
    TENrunning_rewardSTD=[]
    episode_reward_history = []
    running_Variance=[]
    StepReward_history=[]
    TEN_reward_Somma=[]
    #N_upp=0
    ################################################################## End New


    # handle the Ctrl+\ signal to activate/deactivate visualization of the map during the leaning phase [RIC]
    signal.signal(signal.SIGQUIT, changeVisualization)
   
    DQNAgents = []
    for Rango in range(N_robots):
        DQNAgents.append(DQNAgent(seed+Rango))
    


    env = BlobEnv()

    num_actions = 8
    # The first model makes the predictions for Q-values which are used to
    # make a action.

    #input("press enter 1")
    processes = list()
    for robs in range(N_robots): #############################
    
        #t1 = Process(target=DQNAgents[robs].main_Process,args=(num_actions,Children[robs],))#(target=foo, args=(q,))
        t1 = Process(target=DQNAgents[robs].main_Process,
                     args=(num_actions, Children[robs], max_steps_per_episode, robs))
        processes.append(t1)
        t1.start()
        #print("processo partito")
       
  

    #for PLOT
    episode_rewards_log = np.zeros((100000000, 3))
    robot_rewards_log = np.zeros((STEPS_PER_EPISODE, N_robots+1))
    robot_rewards_log[:,0] = range(0, max_steps_per_episode)

    color_map = plt.cm.rainbow
    color_norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
    #print("sono in main prima del while")
    MyEpTest=0;  
    
    while MyEpTest<11:  # Run until solved
        # creo lo sporco random nell'heatmap
        # e la matrice degli ostacoli fissi Wall_Matrix
        #print("sono nel while main, chiamo reset ",N_Walkman_rand)
        MyEpTest=MyEpTest+1
        env.reset()
        #episode_not_zero_percentage_history.clear()
        episode_PercentageSomma_history.clear()
        #del episode_not_zero_percentage_history[:1]
        del episode_PercentageSomma_history[:1]
        #ricevo le posizioni degli agenti


        agent_position=[]
        for robs in range(N_robots):
            agent_position = Parent[robs].recv() #prima comunicazione
            env.update_positionOfRobot_Matrix(agent_position[0],agent_position[1],robs)
        #print("main init pos received")

        #input("press enter 1")
        #print("seconda comunicazione:invio l'env.Data aggiornato al processo",type(env.Data),np.shape(env.Data))
        #invio l'env aggiornato al processo, che si fara la sua copia, 
        # aggiorno la posizione in self.Data[:,:,1] e poi salva lo stato self.state

        for robs in range(N_robots):
            Parent[robs].send(env.Data) #seconda comunicazione
        #print("main init state sent")

        episode_reward = 0
 
        Done_Flag=False

        stored_execution_time = time.time()
        
        ############################################## forth new
        episode_PercentageSomma_history.clear()
        StepReward_history.clear()
        ############################################## end 
        
        countUpdatePeriod=0

        for timestep in range(1, max_steps_per_episode+1):
            #input("press enter 1")
            #print("processes executed in ", time.time() - processes_execution_time)

            if Done_Flag==True:
                Done_Flag=False
                break  

            
            if (timestep%update_period==0) and (countUpdatePeriod<55):

                countUpdatePeriod+=1

           
                env.updateHeatmap(countUpdatePeriod)
                # invio l'env generale aggiornato ai processi 
                #dopo l'applicazione della gaussiana
            
            #print("terza comunicazione:invio l'env.Data aggiornato al processo",type(env.Data),np.shape(env.Data))
            
            else:
                env.updateHeatmap2(countUpdatePeriod)

            
            for robs in range(N_robots): #terza comunicazione
                Parent[robs].send(env.Data)
            #print("main state sent")

            Vect_Done = []
            for robs in range(N_robots): # quarta comunicazione
                Vect_Done.append( Parent[robs].recv() )
            #print("main done received")

            for robs in range(N_robots):
                #print("main received done-",robs,": ",Vect_Done[robs])
                if Vect_Done[robs] == True:
                    #print("MAIN PROCESS ", robs, " ACCOMPLISHED")
                    Done_Flag = True
                    break

            for robs in range(N_robots):  # quinta comunicazione
                Parent[robs].send(Done_Flag)
            #print("main done_flag sent: ", Done_Flag)

            ###########################################################################
            #print("ricevo le posizioni degli agenti e li salvo in una lista")
            #ricevo le posizioni degli agenti
            SommaTot=0;
            SommaBefore=np.sum(env.Data[:,:,0])
            #print("SommaBefore", SommaBefore)
            for robs in range(N_robots):
                agent_position = []
                agent_position = Parent[robs].recv() # sesta comunicazione
                env.update_positionOfRobot_Matrix(agent_position[0], agent_position[1], robs)
            #print("main position received")
            SommaTot=np.sum(env.Data[:,:,0])

            ###########################################################################    
            frame_count += 1

            overall_state = np.zeros((N_totalX,N_totalY,2))
            #input("press enter 1")
            
            Vect_Done_Rew=[]
            #print("quinta comunicazione prima del for")
            for robs in range(N_robots):
                #print("quinta comunicazione dentro del for")
                Vect_Done_Rew.append(Parent[robs].recv()) # settima comunicazione
            #print("quinta comunicazione Vect_Done_Rew",Vect_Done_Rew)
            Vect_Done=[item[0] for item in Vect_Done_Rew]
            #print("Vect_Done ricevuto",Vect_Done)
            #print("quinta comunicazione Vect_Done_Rew ",type(Vect_Done_Rew[0]))
            Vect_Reward=[item[1] for item in Vect_Done_Rew]
            Vect_Episode_reward=[item[2] for item in Vect_Done_Rew]
            #print("main output received")
            StepReward=0
            for robs in range(N_robots):
                #print("Vect_Reward[robs]= ", Vect_Reward[robs])
                # Save the single robot reward into the log (for plotting) [RIC]
                StepReward = StepReward+Vect_Reward[robs]
            StepReward_history.append(StepReward) ## NEW
            #MaxRisk=N_Walkman+env.first_sum
            MaxRisk2=float(N_totalX * N_totalY)-float(np.count_nonzero(env.Wall_Matrix))
            #print("SommaTot,MaxRisk",SommaTot,MaxRisk2)
            PercentageSomma=((MaxRisk2-SommaTot)/MaxRisk2)*100
            #PercentageSomma=(1-SommaTot/MaxRisk2)*100
            #print("SommaTot",SommaTot)
            #PercentageSomma=(SommaBefore-SommaTot)/SommaTot*100
            #non_zero_percentageTOT = 100-( float(np.count_nonzero(env.Data[:, :, 0])) /(float(N_totalX * N_totalY)-float(np.count_nonzero(env.Wall_Matrix))))*100
    
            #episode_not_zero_percentage_history.append(non_zero_percentageTOT)
            episode_PercentageSomma_history.append(PercentageSomma)
            
            ###############################################################################11th block
            with open("Report26"+str(N_robots)+str(N_Walkman)+".txt", "a") as f:
                print(" ",PercentageSomma," ",StepReward, file=f)
            #print(" ciao",PercentageSomma," ciao",StepReward)
            ###############################################################################end
            


            #print("PercentagePulizia,timestep", PercentageSomma,timestep)
            #PLOT Figure 1 - begin
            #print( " Stepreward: ", StepReward, " elapsed: ","PercentagePulizia",PercentageSomma,"timestep",timestep,"non_zero_percentageTOT",non_zero_percentageTOT)
            for robs in range(N_robots):
                #print("Vect_Reward[robs]= ", Vect_Reward[robs])
                # Save the single robot reward into the log (for plotting) [RIC]
                robot_rewards_log[timestep-1, robs+1] = Vect_Reward[robs]

            # PLOT Figure 2 - begin
            if TESTING:
                # plt.clf()
                fig = plt.figure(2)
                fig.clf()

                ax = fig.subplots(nrows=2, ncols=2)  # , figsize=(16, 4))

                # ax1 = fig.add_subplot(221)
                extent = (0, N_totalY, 0, N_totalX)
                ax[0, 0].imshow(env.Data[:, :, 0], cmap=plt.cm.hot, origin='upper', extent=extent)

                ax[0, 0].set_title('State')
                ax[0, 0].set_xlabel('x-cord')
                ax[0, 0].set_ylabel('y-cord')

                # ax2 = fig.add_subplot(222)
                ax[0, 1].imshow(env.Data[:, :, 0], cmap=plt.cm.hot, origin='upper', extent=extent)
                ax[0, 1].imshow(env.Data[:, :, 1], cmap='gray',alpha=0.2, origin='upper', extent=extent)
                ax[0, 1].imshow(env.Wall_Matrix, cmap='copper', alpha=0.2, origin='upper', extent=extent)

                ax[0, 1].set_title('Robots Positions')
                ax[0, 1].set_xlabel('x-cord')
                ax[0, 1].set_ylabel('y-cord')

                # ax3 = fig.add_subplot(223)
                for k in range(N_robots):
                    ax[1, 0].plot(robot_rewards_log[0:timestep, 0],
                                  robot_rewards_log[0:timestep, k + 1], color=color_map(color_norm(k)))

                ax[1, 0].set_title('Robot-specific rewards')
                ax[1, 0].set_xlabel('runs')
                ax[1, 0].set_ylabel('rewards')

                plt.draw()
                plt.pause(0.0001)

            # PLOT Figure 2 - end

            #print("MAIN: check done flag")
            if Done_Flag:
                #print("main restart")
                Done_Flag = False
                #break


        for robs in range(N_robots): #############################
            episode_reward=episode_reward+Vect_Episode_reward[robs]

        episode_reward_history.append(episode_reward)

        
        if len(episode_reward_history) > 1000:
            del episode_reward_history[:1]

        running_reward = np.mean(episode_reward_history)
        
        #running_not_zero_perc=np.mean(episode_not_zero_percentage_history)
        reward_Somma=np.sum(StepReward_history)
        running_Somma=np.mean(episode_PercentageSomma_history)
        max_Somma_history=np.max(episode_PercentageSomma_history)
        min_Somma_history=np.min(episode_PercentageSomma_history)
        
        ######################################################fifth block 
        #testato tutto ok
        running_Variance=np.std(episode_PercentageSomma_history) # Standard deviation della percentuale di pulizia
        running_rewardMean = np.mean(StepReward_history) # media del reward
        running_rewardSTD = np.std(StepReward_history) # Standard deviation del reward
        running_rewardMIN=np.min(StepReward_history)#min reward
        running_rewardMax=np.max(StepReward_history)#max reward

        ###################################################################

        TEN_reward_Somma.append(reward_Somma)
        #max_not_zero_perc=np.max(episode_not_zero_percentage_history)
        #min_not_zero_perc=np.min(episode_not_zero_percentage_history)
        TENepisode_PercentageSomma_history_max.append(max_Somma_history)
        TENepisode_PercentageSomma_history_min.append(min_Somma_history)
        TENepisode_PercentageSomma_history_mean.append(running_Somma)
        ################################################################# sixth block
        TENepisode_PercentageSomma_history_Var.append(running_Variance)
        TENrunning_rewardMean.append(running_rewardMean)
        TENrunning_rewardSTD.append(running_rewardSTD)
        TENrunning_rewardMIN.append(running_rewardMIN)
        TENrunning_rewardMAX.append(running_rewardMax)


        #############################################################################


        
        #del episode_not_zero_percentage_history[:1]
        del episode_PercentageSomma_history[:1]
        
        #episode_not_zero_percentage_history.clear()


        print("N_Walkman",N_Walkman,"episode: ", episode_count, " reward: ", episode_reward, " elapsed: ","PercentagePulizia",running_Somma,"timestep",timestep,
              time.time() - stored_execution_time,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history)
        #print("timestep: ", timestep, ", robot: ", robs, ", history: ", len(rewards_history))
        ######################################################################### 9th block
        print("N_robots",N_robots,"N_Walkman",N_Walkman,"episode: ", episode_count, " reward: ", episode_reward, " elapsed: ","PercentagePulizia",running_Somma,"timestep",timestep, time.time() - stored_execution_time,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history,"running_Variance",running_Variance, "running_rewardMean",running_rewardMean,"running_rewardSTD",running_rewardSTD,"running_rewardMIN",running_rewardMIN,"running_rewardMax",running_rewardMax)
        
        with open("Report26"+str(N_robots)+str(N_Walkman)+".txt", "a") as f:
            print("per ogni Eposodio: N_robots",N_robots,"N_Walkman",N_Walkman,"episode: ", episode_count, " reward: ", episode_reward, " elapsed: ","PercentagePulizia",running_Somma,"timestep",timestep, time.time() - stored_execution_time,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history,"running_Variance",running_Variance, "running_rewardMean",running_rewardMean,"running_rewardSTD",running_rewardSTD,"running_rewardMIN",running_rewardMIN,"running_rewardMax",running_rewardMax,file=f)
        
        #########################################################################end
        stored_execution_time = time.time()

        episode_rewards_log[episode_count, 0] = episode_count
        episode_rewards_log[episode_count, 1] = episode_reward
        episode_rewards_log[episode_count, 2] = running_reward
        '''
        plt.figure(1)
        plt.clf()

        plt.plot(episode_rewards_log[0:episode_count, 0],
                 episode_rewards_log[0:episode_count, 1], 'b')

        plt.plot(episode_rewards_log[0:episode_count, 0],
                 episode_rewards_log[0:episode_count, 2], 'r')

        plt.draw()
        plt.pause(0.0001)
        #PLOT Figure 1 - end
        '''
        #for robs in range(N_robots):
        #    DQNAgents[robs].Check_condition_Solving()


        my_task_solved=False

        for robs in range(N_robots):  # settima comunicazione
            Parent[robs].send([robs,my_task_solved])
        episode_count += 1
    ########################################seventh block    
    Tot_running_Somma=np.mean(TENepisode_PercentageSomma_history_mean)
    Tot_min_Somma_history=np.min(TENepisode_PercentageSomma_history_min)
    Tot_max_Somma_history=np.max(TENepisode_PercentageSomma_history_max)
    Tot_running_SommaSTD=np.mean(TENepisode_PercentageSomma_history_Var)
    
    Tot_episode_rewardMean = np.mean( TEN_reward_Somma) # media del reward di episodio
    Tot_episode_rewardSTD = np.std( TEN_reward_Somma) # Standard deviation del reward di episodio
    Tot_episode_rewardMIN=np.min( TEN_reward_Somma)#min reward di episodio
    Tot_episode_rewardMax=np.max( TEN_reward_Somma)#max reward di episodio

    Tot_rew_Mean=np.mean(TENrunning_rewardMean)
    Tot_rew_STD=np.mean(TENrunning_rewardSTD)
    Tot_rew_min=np.min(TENrunning_rewardMIN)
    Tot_rew_max=np.max(TENrunning_rewardMAX)
    ############################################################# end
    print("TENepisode_PercentageSomma_history_max",TENepisode_PercentageSomma_history_max)
    print("N_robots",N_robots,"N_Walkman",N_Walkman," reward: ", episode_reward,"running_reward",running_reward, " elapsed: ","PercentagePulizia",running_Somma,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history)
    
    ##############################################################################################8th block
    with open("Report26"+str(N_robots)+str(N_Walkman)+".txt", "a") as f:
        print("Fine 50 Episodi: N_robots",N_robots,"N_Walkman",N_Walkman,"Tot_episode_rewardMax",Tot_episode_rewardMax,"Tot_episode_rewardMIN",Tot_episode_rewardMIN,"Tot_episode_rewardSTD",Tot_episode_rewardSTD,"Tot_episode_rewardMean",Tot_episode_rewardMean,"episode_reward: ", episode_reward,"running_reward",running_reward,"Tot_rew_max",Tot_rew_max,"Tot_rew_min",Tot_rew_min,"Tot_rew_STD",Tot_rew_STD,"Tot_rew_Mean",Tot_rew_Mean,"Tot_running_Somma",Tot_running_Somma,"Tot_running_SommaSTD",Tot_running_SommaSTD,"Tot_min_Somma_history",Tot_min_Somma_history,"Tot_max_Somma_history",Tot_max_Somma_history,file=f)   
        #print("FINE 50 Episodi ",Tot_rew_max," ",Tot_rew_min," ",Tot_rew_STD," ",Tot_rew_Mean," ",Tot_running_Somma," ",Tot_running_SommaSTD," ",Tot_min_Somma_history," ",Tot_max_Somma_history, file=f)
        
    ##########################################################################################################
    episode_PercentageSomma_history.clear()
    ############################################### new second block
    #testato il secondo blocco, tutto ok
    TENepisode_PercentageSomma_history_max.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_min.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_mean.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_Var.clear()
    StepReward_history.clear()
    TENrunning_rewardMIN.clear()
    TENrunning_rewardSTD.clear()
    TENrunning_rewardMean.clear()
    TENrunning_rewardMAX.clear()
    TEN_reward_Somma.clear()
    ##########################################################
    for robs in range(N_robots):
        Parent[robs].close()
        Children[robs].close()

    for robs in range(N_robots):
        processes[robs].join()

if __name__ == "__main__":

    main()
