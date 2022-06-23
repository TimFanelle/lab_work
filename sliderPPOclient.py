import torch
import torch.nn as nn
from gym import spaces

from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network import FeedForwardNN
import numpy as np

import socket

class singleFinger():
    def __init__(self, motors, encoders):
        # Define dimensions
        mots = np.array([0]*motors)
        mote = np.array([1]*motors)
        self.action_space = spaces.Box(low=mots, high=mote, dtype=np.float64)
            # I might also need to make a spaces.box for the observation space
        self.obs_dim = encoders 
        self.act_dim = self.action_space.shape[0]

        #Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.__init_hyperparameters()

        #initialize optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)

        # Create varible for the matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.s = socket.socket()
        self.host = "127.0.0.1"
        self.port = 4268
    
    
    def __init_hyperparameters(self):
        self.max_timesteps_per_episode = 1000

        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

        self.goal_volt = 1
        self.last_volt = 0
    

    def get_cur_obs(self):
        obs = [0] * self.obs_dim

        #TODO: read all the encoder states and the slider state
        '''
        # this is stand-in pseudocode that will have to be changed
        for i in range(len(encoders)):
            obs[i] = encoders[i].readout()
        obs[-1] = switch.state()
        '''
        self.s.connect((self.host, self.port))
        self.s.send("2;".encode())
        retString = self.s.recv(1024).decode()
        self.s.close()
        rets = retString.split(", ")
        for i in range(len(rets)):
            obs[i] = float(rets[i])
        
        obs = np.array(obs)
        return obs
    

    def step(self, action):
        # take input from action space and do muscle movements
        # TODO: write this
        action = action.detach().numpy()
        '''
        # this is stand-in pseudocode that will have to be changed
        for i in range(len(action)):
           motors[i].write(action[i]) #this should be done for 0.01 seconds
        '''
        sendString = "1;"
        for i in range(len(action)):
            sendString += str(action[i])
            if(i != len(action)-1):
                sendString += ";"
        
        self.s.connect((self.host, self.port))
        self.s.send(sendString.encode())
        retString = self.s.recv(1024).decode()
        self.s.close()
        rets = retString.split(", ")
        for i in range(len(rets)):
            rets[i] = float(rets[i])

        # get observations
        observed = rets[:-1] #self.get_cur_obs()

        # measure input from slider switch
        voltage_inp = rets[-1]

        # determine if done
        done = bool(voltage_inp >= self.goal_volt-0.1) # subtract 0.1 to give some leeway and account for fluctuations

        reward = 0
        if done:
            reward = 100.0
        delta_volt = voltage_inp - self.last_volt
        reward += (delta_volt**2) * np.sign(delta_volt)
        return observed, reward, done


    def rollout(self):
        #episode data
        ep_obs = []
        ep_acts = []
        ep_log_probs = []
        ep_rews = []
        ep_rtgs = []

        # number of timesteps
        t = 0
        obs = self.get_cur_obs()
        done = False

        for ep_t in range(self.max_timesteps_per_episode):
            #increment timesteps
            t += 1

            #collect observation
            ep_obs.append(obs)

            action, log_prob = self.get_action(obs)
            obs, rew, done = self.step(action)

            #collect reward, action, and log_prob
            ep_rews.append(rew)
            ep_acts.append(action)
            ep_log_probs.append(log_prob)

            if done:
                break
        t += 1 # plus 1 because timestep started at 0

        #reshape data as tensors
        ep_obs = torch.tensor(ep_obs, dtype=torch.float)
        ep_acts = torch.tensor(ep_acts, dtype=torch.float)
        ep_log_probs = torch.tensor(ep_log_probs, dtype=torch.float)

        ep_rtgs = self.compute_rtgs(ep_rews)

        #return data
        return ep_obs, ep_acts, ep_log_probs, ep_rtgs, t
    

    def compute_rtgs(self, ep_rews):
        ep_rtgs = []

        #iterate through the episode backwardsto maintain order
        discounted_reward = 0
        for rews in reversed(ep_rews):
            discounted_reward = rews + discounted_reward*self.gamma
            ep_rews.insert(0, discounted_reward)
        #convert rewards-to-go into a tensor
        ep_rtgs = torch.tensor(ep_rtgs, dtype=torch.float)

        return ep_rtgs
    

    def get_action(self, obs):
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # create multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # sample an action from the distribution and get its log probability
        action = dist.sample()
        action = torch.tensor(np.clip(action.detach().numpy(), self.action_space.low, self.action_space.high), dtype=torch.float)
        log_prob = dist.log_prob(action)

        # return the action and log_prob
        return action.detach().numpy(), log_prob.detach()
    

    def evaluate(self, ep_obs, ep_acts):
        # Query critic network for a value for each observation
        V = self.critic(ep_obs).squeeze()

        # calculate the log probabilityes of the episode actions using the most up-to-date actor network
        mean = self.actor(ep_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(ep_acts)

        # return predicted values and log probabilities
        return V, log_probs
    

    def learn(self, total_timesteps):
        t_so_far = 0 # timesteps so far
        while t_so_far < total_timesteps:
            ep_obs, ep_acts, ep_log_probs, ep_rtgs, ep_len = self.rollout()

            # add how many timesteps were taken in the episode to the total so far
            t_so_far += np.sum(ep_len)

            # calcule V_{phi, k}
            V, _ = self.evaluate(ep_obs, ep_acts)

            #calculate advantage
            A_k = ep_rtgs - V.detach()

            # normalize advantages
            A_k = (A_k - A_k.mean())/(A_k.std()*1e-10)

            for _ in range(self.n_updates_per_iteration):
                # calculate Pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(ep_obs, ep_acts)

                # calculate ratios
                ratios = torch.exp(curr_log_probs - ep_log_probs)

                # calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1+ self.clip)

                actor_loss = (-torch.min(surr1, surr2))
                critic_loss = nn.MSELoss()(V, ep_rtgs)

                # calculate gradients and perform backpropagation on the actor and critic networks
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            # Wait for the setup to be reset
            input("Please reset setup and press Enter to continue")
            input("Please press enter to confirm continuation")
    
    def endIt(self):
        # Sends a message to the server to shutdown
        self.s.connect((self.host, self.port))
        self.s.send("3;".encode())
        lrs = self.s.recv(1024).decode()
        self.s.close()
        print(lrs)
        print("Server ended successfully")
    
num_tendons = 6
num_encoders = 6
model = singleFinger(num_tendons, num_encoders)
model.learn(10000)

model.endIt()
