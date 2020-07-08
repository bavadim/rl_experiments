#!/usr/bin/env python3

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


env = gym.make('CartPole-v1')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)
    
policy = Policy()
if torch.cuda.is_available():
    policy.cuda()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

def select_action(observation):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.tensor(observation, dtype=torch.float)
    
    probs = policy(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    
    return action, m.log_prob(action)

def put_reward(action, prob, reward):
    loss = -prob * reward / 100
    loss.backward()
    
def update_policy():
    optimizer.step()
    optimizer.zero_grad()

def main(episodes):
    running_reward = 0
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
    
        global_reward = 0
        for time in range(1000):
            action, prob = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.item())

            global_reward += reward
            # Save reward
            put_reward(action, prob, reward)
            #env.render()
            if done:
                break
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward + global_reward) / 2

        if episode % 1000 == 0:
            update_policy()
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break
            
    env.close()
    torch.save(policy, 'models/')


main(episodes = 1000000)