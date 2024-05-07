from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from tqdm import tqdm

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		print('Evaluating...')
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			self.logger.video.init(self.env, enabled=self.cfg.save_video and (i < self.cfg.num_videos))
			while not done:
				if self.cfg.perturb:
					obs = self.perturb_obs(obs)

				action = self.agent.act(obs, t0=t==0, eval_mode=True).detach()

				if self.cfg.perturb:
					action = self.perturb(action)

				obs, reward, done, info = self.env.step(action)
				
				ep_reward += reward
				t += 1
				self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict({k: v.unsqueeze(0) for k,v in obs.items()}, batch_size=(1,)).cpu()
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.empty_like(self.env.rand_act())
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td				

	def update_perturb(self):
		"""Get perturbation factor for action space."""
		print("Updating perturbation factor... Perturb scale:", self.cfg.perturb_scale)
		self.perturb_factor = self.cfg.perturb_scale * torch.randn(self.env.action_space.shape[0])
		print("Perturb factor:", self.perturb_factor.numpy())

	def action_skip(self):
		if self.cfg.action_skip:
			return self._step % self.cfg.action_skip == 0
		else: 
			return 1

	def perturb(self, action):
		"""Perturb action space."""
		return action * (1. + self.perturb_factor + self.cfg.action_noise * torch.randn_like(action)) * self.action_skip()
	
	def perturb_obs(self, obs):
		"""Perturb observation space."""
		return obs + self.cfg.obs_noise * torch.randn_like(obs)

	def set_env(self):
		
		phys = self.env.unwrapped.physics
		phys.named.model.geom_size['target', 0] = self.cfg.target_size 
		phys.named.model.geom_margin['target'] = self.cfg.target_margin 
		phys.set_state(np.array(self.cfg.init_state)) # fixed initial conditions
		# fixed target position
		angle = self.cfg.target[0]
		radius = self.cfg.target[1]
		phys.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
		phys.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True

		self.set_env()

		# Perturbation factor for action space
		# Random scaling, shape of action space
		if self.cfg.perturb:
			self.update_perturb()

		while self._step <= self.cfg.steps:

			if self._step % 100 == 0:
				print("Step:", self._step)

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if self._step > 0:
					ep_len = len(self._tds)
					print("episode length:", ep_len)
					
					# adaptive control
					if self.cfg.control:
						# for _ in range(self.cfg.ctrl_update_steps): # can't repeat online learning
						_train_metrics = self.agent.control_update(torch.cat(self._tds))
						train_metrics.update(_train_metrics)

					# Log episode
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					data = torch.cat(self._tds).detach().shape
					# print(self.buffer._storage._max_size_along_dim0(data))
					# print("maxsize:", self.buffer._storage.max_size)
					# print(data)
					self._ep_idx = self.buffer.add(torch.cat(self._tds).detach())

					# Update agent
					if self._step >= self.cfg.seed_steps:
						if self._step == self.cfg.seed_steps:
							num_updates = int(self.cfg.updates_per_step * self.cfg.seed_steps)
							print('Pretraining agent on seed data...')
							# num_updates = 10
						else:
							# num_updates = ep_len
							num_updates = int(self.cfg.updates_per_step * ep_len)
						for _ in range(num_updates):
							_train_metrics = self.agent.update(self.buffer)
						train_metrics.update(_train_metrics)

				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()


			if self.cfg.perturb:
				effective_action = self.perturb(action)
			else:
				effective_action = action

			obs, reward, done, info = self.env.step(effective_action.detach())
			if self.cfg.perturb:
				obs = self.perturb_obs(obs)
			self._tds.append(self.to_td(obs, action, reward))

			if self.cfg.perturb and (self._step >= self.cfg.seed_steps) and \
			(self._step % self.cfg.perturb_steps == 0):
				self.update_perturb()

			self._step += 1
	
		self.logger.finish(self.agent)
