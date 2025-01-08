import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel


class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.model = WorldModel(cfg).to(self.device)

		if cfg.lr_factor:
			lr_factor = np.sqrt(cfg.batch_size / 512)
			cfg.lr *= lr_factor
			cfg.pi_lr *= lr_factor
			cfg.lr_ctrl_dyn *= lr_factor
			cfg.lr_ctrl_pi *= lr_factor

		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters(), 'lr': self.cfg.dyn_lr},
			{'params': self.model._dynamics_z_only.parameters(), 'lr': self.cfg.dyn_lr},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.pi_lr, eps=1e-5)
		
		self.ctrl_dyn_optim = torch.optim.Adam([
			{'params': self.model._ctrl_dynamics.parameters()},
		], lr=self.cfg.lr_ctrl_dyn)

		if self.cfg.ctrl_opt == 'adam':
			self.ctrl_pi_optim = torch.optim.Adam(self.model._ctrl_pi.parameters(), lr=self.cfg.lr_ctrl_pi, eps=1e-5)
		elif self.cfg.ctrl_opt == 'sgd':
			self.ctrl_pi_optim = torch.optim.SGD(self.model._ctrl_pi.parameters(), lr=self.cfg.lr_ctrl_pi, momentum=0.9)
		else:
			raise ValueError("Invalid control optimiser")

		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)

		self.latent_state = torch.zeros(cfg.latent_dim, device=self.device)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=torch.device('cpu'))
		self.model.load_state_dict(state_dict["model"], strict=False)

	# @torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, ctrl=False, actions=None):
		"""
		Select an action by planning in the latent space of the world model.
		
		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)

		if not self.cfg.input_delay_buffer:
			z = self.model.encode(obs, task)
		else:
			# print(obs.shape, actions.shape)
			z = self.model.encode_delayed(obs, actions.detach(), task)

		if self.cfg.rec_latent and not t0:
			z = (1-self.cfg.rec_alpha)*z + (self.cfg.rec_alpha)*self.latent_state.detach()

		if self.cfg.mpc:
			a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		else:
			a = self.model.pi(z, task)[int(not eval_mode)][0]

		if self.cfg.rec_latent:
			self.latent_state = self.model.next(z, a.unsqueeze(0), task)[0].detach()

		if ctrl:
			if self.cfg.ctrl_full:
				a_ctrl = self.model._ctrl_pi(z.detach())[0]
				a_total = a + a_ctrl.detach()

			else:
				# get output of last hidden layer of self.model._pi(z, task):
				last_hidden = self.model._pi[:-1](z)
				a_ctrl = self.model._ctrl_pi(last_hidden.detach())[0]
				# a_ctrl = self.model._ctrl_pi(z)[0] # full control
				a_total = a + a_ctrl.detach() # don't backpropagate through ctrl_pi

			return a_total.cpu(), a.cpu(), a_ctrl.cpu()

		return a.cpu(), a.cpu(), torch.zeros_like(a).cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)[0]
			G += discount * reward
			discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""		
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)[0]
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for i in range(self.cfg.iterations):

			# Sample actions
			actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		a, std = actions[0], std[0]
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a.clamp_(-1, 1)
		
	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type='avg')

		# transfered cost to reward in training
		# here works better, but for mpc doesn't work
		if not self.cfg.cost_in_reward:

			if self.cfg.cost_thres:
				thres_action = (pis.abs() - self.cfg.cost_thres).clamp(min=0) * (1/(1 - self.cfg.cost_thres))
			else:
				thres_action = pis
		
			qs -= 1 * self.cfg.control_cost * (thres_action**2).sum(dim=-1).unsqueeze(-1)
		
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample()

		action = action.detach() # don't backprop through action here.

		# print(obs.shape, action.shape, reward.shape)

		# Compute targets
		with torch.no_grad():
			if not self.cfg.input_delay_buffer:
				next_z = self.model.encode(obs[1:], task)	
				td_targets = self._td_target(next_z, reward, task)
			else:
				# print(obs.shape, action.shape)
				# print('update with delay buffer')
				next_z = self.model.encode_delayed(obs[1+self.cfg.input_delay_buffer:], action[1:].detach(), task)
				action_start = action[:self.cfg.input_delay_buffer]
				action = action[self.cfg.input_delay_buffer:]
				obs = obs[self.cfg.input_delay_buffer:]
				reward = reward[self.cfg.input_delay_buffer:]
				td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		if not self.cfg.input_delay_buffer:
			z = self.model.encode(obs[0], task)
		else:
			z = self.model.encode_delayed(obs[0], action_start.detach(), task)[0]
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z, sig = self.model.next(z, action[t], task)
			if self.cfg.ctrl_mse:
				consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t 
			else:
				consistency_loss += torch.nn.GaussianNLLLoss()(z, next_z[t], sig) * self.cfg.rho**t
			zs[t+1] = z
			
		if self.cfg.ctrl_mse and self.cfg.learn_sigma:
			consistency_loss += torch.nn.GaussianNLLLoss()(z.detach(), next_z[-1].detach(), sig) 

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		
		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		##################
		# Update _dynamics_z_only
		if self.cfg.two_stage:
			self.optim.zero_grad(set_to_none=True)
			z = self.model.encode(obs[0], task).detach()
			consistency_loss_z = 0
			for t in range(self.cfg.horizon):
				z, sig = self.model.z_only_pred(z, task)
				if self.cfg.ctrl_mse:
					consistency_loss_z += F.mse_loss(z, next_z[t].detach()) * self.cfg.rho**t 
				else:
					consistency_loss_z += torch.nn.GaussianNLLLoss()(z, next_z[t].detach(), sig) * self.cfg.rho**t

			if self.cfg.ctrl_mse and self.cfg.learn_sigma:
				consistency_loss_z += torch.nn.GaussianNLLLoss()(z.detach(), next_z[-1].detach(), sig) 

			consistency_loss_z *= (1/self.cfg.horizon)
			consistency_loss_z.backward()
			self.optim.step()
		else:
			consistency_loss_z = torch.zeros(1)
		##################

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
			"consistency_loss_z": float(consistency_loss_z.mean().item())
		}
	
	# def control_predict_old(self, obs, action, task=None, diff=True):
	# 	obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
	# 	action = action.to(self.device, non_blocking=True).unsqueeze(0)
	# 	zs_obs = self.model.encode(obs, task).detach()
	# 	zs_pred = self.model.ctrl_pred(zs_obs, action, task)
	# 	z_mu, z_sig2 = zs_pred[:,:self.cfg.latent_dim], zs_pred[:,self.cfg.latent_dim:].exp()
	# 	if diff:
	# 		z_mu = z_mu + zs_obs
	# 	return z_mu[0].detach().cpu().numpy(), z_sig2[0].detach().cpu().numpy()
	
	# uses _dynamics as forward model
	def control_update(self, episode, pred_only=False): #diff=True
		"""
		Adpative control update function.
		Learns control policy, uses _dynamics as forward model.
		
		Args:
			episode (dict): Episode sample. Formatted as from the replay buffer. 
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		episode = episode.float().to(self.device, non_blocking=True)

		obs, action, reward = episode['obs'], episode['action'][1:], episode['reward'][1:]
		task = None
		# print(obs.shape, action.shape, reward.shape)

		# print("Episode length:", ep_len)

		# Prepare for update
		self.ctrl_pi_optim.zero_grad(set_to_none=True) 
		self.pi_optim.zero_grad(set_to_none=True) # get these gradient, then transfer to self.model._ctrl_pi
		# self.ctrl_dyn_optim.zero_grad(set_to_none=True) # learn only policy here.
		self.model.train()

		# Compare latent predictions
		if not self.cfg.input_delay_buffer:
			zs_obs = self.model.encode(obs, task).detach()
		else:
			# print(obs.shape, action.shape)
			zs_obs = self.model.encode_delayed(obs[self.cfg.input_delay_buffer:], action.detach(), task).detach()

			action = action[self.cfg.input_delay_buffer:]

		a_len = len(action)
		ep_len = len(zs_obs)

		H = self.cfg.ctrl_horizon
		Z = self.cfg.latent_dim
		zs_pred = zs_obs[:-H].detach()

		aux_grad = torch.zeros_like(action, requires_grad=True)
		if self.cfg.ctrl_full:
			a_pred = action.detach() + aux_grad#.detach().requires_grad_()
		else:
			a_pred = action


		# learn only first step action
		# zs_pred, sig = self.model.next(zs_pred, a_pred[:a_len-H+1], task)
		# for i in range(1,H):
		# 	# print(zs_pred.shape, a_pred[i:a_len-H+i+1].shape, task)
		# 	zs_pred, sig = self.model.next(zs_pred, a_pred[i:a_len-H+i+1].detach(), task)
		
		consistency_loss = 0
		for i in range(H):
			# print(zs_pred.shape, a_pred[i:a_len-H+i+1].shape, task)
			zs_pred, sig = self.model.next(zs_pred, a_pred[i:a_len-H+i+1], task)
			# zs_pred, sig = self.model.ctrl_pred(zs_pred, a_pred[i:a_len-H+i+1], task) # trying again separate FM

			target = zs_obs[i+1:ep_len-H+i+1]

			if self.cfg.ctrl_rec_loss:
				if self.cfg.ctrl_mse:
					consistency_loss += F.mse_loss(zs_pred, target) * self.cfg.rho**i
					sig = torch.zeros(1)
				else: 
					loss = torch.nn.GaussianNLLLoss()
					consistency_loss += loss(zs_pred, target, sig) * self.cfg.rho**i
			

		if pred_only:
			return zs_pred, sig

		# if diff: # was wrong
		# target = zs_obs[H:] - zs_obs[:-H]
		# else:

		if not self.cfg.ctrl_rec_loss:
			target = zs_obs[H:]

			if self.cfg.ctrl_mse:
				consistency_loss = F.mse_loss(zs_pred, target) 
				sig = torch.zeros(1)
			else: 
				loss = torch.nn.GaussianNLLLoss()
				consistency_loss = loss(zs_pred, target, sig)
		
		# consistency_loss *= (1/(ep_len - self.cfg.ctrl_horizon - 1))
		consistency_loss *= (1/self.cfg.ctrl_horizon)

		# Update model
		consistency_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

		# Backward to policy weights
		# action.backward(a_pred.grad, retain_graph=True)

		if (not self.cfg.ctrl_full) and (self.model._pi[-1].weight.grad is not None):

			# if self.cfg.action_weighting:
			# 	print(self.model._ctrl_pi.weight.grad.shape)
			# 	self.model._ctrl_pi.weight.grad =  multiply grad by action vector abs value

			# else:
				
			# update self.model._ctrl_pi instead:
			self.model._ctrl_pi.weight.grad = -self.model._pi[-1].weight.grad[:self.cfg.action_dim].clone()
			# TODO: add bias?

		elif self.cfg.ctrl_full:
			# a_ctrl = self.model._ctrl_pi(zs_obs[:-1])
			a_ctrl = episode['action_ctrl'][1+self.cfg.input_delay_buffer:]
			a_ctrl.backward(gradient=-aux_grad.grad)
		else:
			print("No gradient for action policy")
			
		self.ctrl_pi_optim.step() 

		# print("ctrl pi norms:", self.model._ctrl_pi.weight.norm(), self.model._ctrl_pi.bias.norm())

		self.ctrl_dyn_optim.step()

		# Return training statistics
		self.model.eval()
		return {
			"ctrl_loss": float(consistency_loss.mean().item()),
			"ctrl_sig": sig.mean().sqrt().item(),
			"ctrl_dist": (zs_pred[:self.cfg.dist_range,:2]-target[:self.cfg.dist_range,:2]).norm(dim=1).mean()
			# "grad_norm_ctrl": float(grad_norm)
		}
	
# builds separate forward model
	# def control_update_old(self, episode, diff=True):
	# 	"""
	# 	Adpative control update function.
		
	# 	Args:
	# 		episode (dict): Episode sample. Formatted as from the replay buffer. 
		
	# 	Returns:
	# 		dict: Dictionary of training statistics.
	# 	"""
	# 	episode = episode.to(self.device, non_blocking=True)

	# 	obs, action, reward = episode['obs'], episode['action'][1:], episode['reward'][1:]
	# 	ep_len = len(obs)
	# 	task = None
	# 	# print(obs.shape, action.shape, reward.shape)

	# 	# print("Episode length:", ep_len)

	# 	# Prepare for update
	# 	self.ctrl_pi_optim.zero_grad(set_to_none=True)
	# 	self.pi_optim.zero_grad(set_to_none=True) # get these gradient, then transfer to self.model._ctrl_pi
	# 	self.ctrl_dyn_optim.zero_grad(set_to_none=True)
	# 	self.model.train()

	# 	# Compare latent predictions
	# 	zs_obs = self.model.encode(obs, task).detach()
	# 	zs_pred = self.model.ctrl_pred(zs_obs[:-1], action, task)[:-self.cfg.ctrl_horizon+1]
	# 	mu, sig = zs_pred[:,:self.cfg.latent_dim], zs_pred[:,self.cfg.latent_dim:].exp()

	# 	if diff: # todo: check this, in ctrl_pred()
	# 		target = zs_obs[self.cfg.ctrl_horizon:] - zs_obs[:-self.cfg.ctrl_horizon]
	# 	else:
	# 		target = zs_obs[self.cfg.ctrl_horizon:]

	# 	if self.cfg.ctrl_mse:
	# 		consistency_loss = F.mse_loss(mu, target) 
	# 		sig = torch.zeros(1)
	# 	else: 
	# 		loss = torch.nn.GaussianNLLLoss()
	# 		consistency_loss = loss(mu, target, sig)
		
	# 	# consistency_loss *= (1/(ep_len - self.cfg.ctrl_horizon - 1))

	# 	# Update model
	# 	consistency_loss.backward()
	# 	# grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

	# 	if (self.model._pi[-1].weight.grad is not None):
	# 		# update self.model._ctrl_pi instead:
	# 		self.model._ctrl_pi.weight.grad = -self.model._pi[-1].weight.grad[:self.cfg.action_dim].clone()

	# 		self.ctrl_pi_optim.step() 

	# 	# print("ctrl pi norms:", self.model._ctrl_pi.weight.norm(), self.model._ctrl_pi.bias.norm())

	# 	self.ctrl_dyn_optim.step()

	# 	# Return training statistics
	# 	self.model.eval()
	# 	return {
	# 		"ctrl_loss": float(consistency_loss.mean().item()),
	# 		"ctrl_sig": sig.mean().sqrt().item(),
	# 		"ctrl_dist": (mu[:self.cfg.dist_range,:2]-target[:self.cfg.dist_range,:2]).norm(dim=1).mean()
	# 		# "grad_norm_ctrl": float(grad_norm)
	# 	}
	