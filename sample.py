#!/usr/bin/env python3
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from typing import List, Type

class Agent:
	def act(self, observation):
		pass

	def observe_results(reward, done):
		pass

class RandomAgent(Agent):
	"""The world's simplest agent!"""
	def __init__(self, action_space):
		self.action_space = action_space

	def act(self, observation):
		return self.action_space.sample()

	
learning_rate = 0.01
gamma = 0.99

class ReinforceAgent(Agent):

	class Policy(nn.Module):
		def __init__(self, env):
			super(ReinforceAgent.Policy, self).__init__()
			self.state_space = env.observation_space.shape[0]
			self.action_space = env.action_space.n
			
			self.l1 = nn.Linear(self.state_space, 16, bias=False)
			self.l2 = nn.Linear(16, self.action_space, bias=False)
			
			self.gamma = gamma
			
			# Episode policy and reward history 
			
			self.policy_history = torch.tensor(0)
			self.reward_episode = []
			
			# Overall reward and loss history
			self.loss_history = []
		
		def forward(self, x):    
			return nn.functional.softmax(self.l2(nn.functional.relu(nn.functional.dropout(self.l1(x), p=0.6))), dim=-1)

	"""The world's simplest agent!"""
	def __init__(self, env):
		self.policy = ReinforceAgent.Policy(env)
		self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

	def select_action(self, observation: np.array) -> int:
		state = torch.from_numpy(observation).type(torch.FloatTensor)
		state = self.policy(state)
		c = Categorical(state)
		action = c.sample()
		
		# Add log probability of our chosen action to our history    
		if self.policy.policy_history.dim() != 0:
			self.policy.policy_history = torch.cat([self.policy.policy_history, c.log_prob(action)])
		else:
			self.policy.policy_history = (c.log_prob(action))
		return action
		
	def update_policy(self):
		R = 0
		rewards = []
		
		# Discount future rewards back to the present using gamma
		for r in self.policy.reward_episode[::-1]:
			R = r + gamma * R
			rewards.insert(0,R)
			
		# Scale rewards
		rewards = torch.FloatTensor(rewards)
		#rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
		#print(rewards)
		
		# Calculate loss
		#loss = (torch.sum(torch.mul(self.policy.policy_history, rewards).mul(-1), -1))

		loss = -torch.sum(self.policy.policy_history) * torch.sum(rewards)
		
		# Update network weights
		self.optimizer.zero_grad()
		print(self.policy.l1)
		loss.backward()
		self.optimizer.step()
		
		#Save and intialize episode history counters
		self.policy.loss_history.append(loss)
		self.policy.policy_history = torch.tensor(0)
		
		res_rew = np.sum(self.policy.reward_episode)

		self.policy.reward_episode = []

		return res_rew, loss.item()

	def act(self, observation) -> int:
		return self.select_action(observation).item()

	def observe_results(self, reward, done):
		self.policy.reward_episode.append(reward)
		if done:
			return self.update_policy()
		else:
			return None, None


class Experiment(object):
	def __init__(self, envName: str, agent: Type[Agent]):
		self.env = gym.make(envName)
		self.agent = agent

	def run(self):
		res = []
		agent = self.agent(self.env)
		for i_episode in range(200000000):
			observation = self.env.reset()
			t = 1
			while True:
				#self.env.render()
				action = agent.act(observation)
				observation, reward, done, info = self.env.step(action)
				r, l = agent.observe_results(reward, done)
				if done:
					if i_episode % 100 == 0:
						print(f"Episode {i_episode} finished after {t} timesteps, {r} {l}")
					break
				t += 1
			res.append(t)
		self.env.close()
		return res


if __name__ == '__main__':
	Experiment('CartPole-v1', ReinforceAgent).run()