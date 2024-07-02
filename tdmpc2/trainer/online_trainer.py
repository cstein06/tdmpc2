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
		ep_rewards, ep_successes, ep_lens = [], [], []
		ep_tds = []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			# _tds = [self.to_td(obs)]
			_tds = []
			
			if i < 2 or self.cfg.fixed_init_state:
				obs = self.set_init_state()

			while not done:
				if self.cfg.perturb:
					obs_pert = self.perturb_obs(obs)

				action, action_orig, action_ctrl = self.agent.act(obs_pert, t0=t==0, eval_mode=True, ctrl=self.cfg.control)
				action = action.detach()
				action_orig = action_orig.detach()
				action_ctrl = action_ctrl.detach()

				if self.cfg.perturb:
					action_eff = self.perturb(action, obs=obs)
					
				new_obs, reward, done, info = self.env.step(action_eff)
				
				_tds.append(self.to_td(obs, action_orig, reward,
								action_eff=action_eff, action_ctrl=action_ctrl,
								action_total=action))
				obs = new_obs

				ep_reward += reward
				t += 1
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lens.append(t)
			_tds = torch.cat(_tds)
			ep_tds.append(_tds)

			if self.cfg.save_video and (i < self.cfg.num_videos):
				self.logger.video.init(self.env, enabled=True)

				self.env.reset()

				for t_ in range(0, t):

					phys = self.env.unwrapped.physics
					phys.set_state(_tds[t_]["obs"].numpy())

					_, _, done, _ = self.env.step(torch.zeros_like(action))

					if done:
						break

					if self.cfg.control:
						# print("position:", obs[:2])
						# print("action:", action)
						# print("action_eff:", action_eff)
						# print("action_ctrl:", action_ctrl)
						if t_ < t-self.cfg.ctrl_horizon:
							future_pos = _tds[t_+self.cfg.ctrl_horizon]["obs"][:2]
						else:
							future_pos = None

						pred = self.show_prediction(obs=_tds[t_]["obs"], action=_tds[t_]["action"], action_eff=_tds[t_]["action_eff"], action_ctrl=_tds[t_]["action_ctrl"], action_total=_tds[t_]["action_total"], future_obs=future_pos)

						# if t % 50 == -1:
						# 	print("Observation:", obs)
						# 	print("Prediction:", pred)
							
					self.logger.video.record(self.env)

				self.logger.video.save(self._step, key=f'videos/eval_video_{i+1}')

		mid_point = (np.array(self.cfg.target) + self.cfg.init_state[:2]) / 2.
		# deviation = (ep_tds[0]["obs"][:,:2]**2).sum(axis=0).min().item() # deviation from origin
		deviation = np.linalg.norm(ep_tds[0]["obs"][:,:2] - mid_point, axis=1).min().item() # deviation from midpoint
		print("Deviation:", deviation)

		# write log_episodes to file and save
		torch.save(ep_tds, self.cfg.work_dir / 'log_episodes.npy')

		# save as wandb artifact
		self.logger.log_wandb_artifact(str(self.cfg.work_dir / f'log_episodes.npy'), f'log_episodes_{self._step}.npy')
	
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length=np.nanmean(ep_lens),
			deviation=deviation,
		)

	def to_td(self, obs, action=None, reward=None, **kwargs):
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
		if kwargs:
			for k,v in kwargs.items():
				td[k] = v.unsqueeze(0)
		return td				

	def update_perturb(self):
		"""Get perturbation factor for action space."""
		print("Updating perturbation factor... Perturb scale:", self.cfg.perturb_scale)
		# self.perturb_factor = self.cfg.perturb_scale * torch.randn(self.env.action_space.shape[0]) #  random
		# self.perturb_factor = self.cfg.perturb_scale * torch.tensor([-1.,1.]) * (((self._step / self.cfg.perturb_steps) % 5) - 2) / 2.

		# seq = [torch.tensor([-1.,1.]), torch.tensor([1.,-1.])]
		# seq = [torch.tensor([-1.,1.])]
		seq = [torch.tensor([0.,0.]), torch.tensor([-1.,1.])]
		step = (self._step // self.cfg.perturb_steps) % len(seq)
		self.perturb_factor = self.cfg.perturb_scale * seq[step]
		print("Perturb factor:", self.perturb_factor.numpy())

	def action_skip(self):
		if self.cfg.action_skip:
			return self._step % self.cfg.action_skip == 0
		else: 
			return 1

	def perturb(self, action, obs=None):
		"""Perturb action space."""
		action_pert = action * (1. + self.perturb_factor + self.cfg.action_noise * torch.randn_like(action)) * self.action_skip()

		if self.cfg.force_field:
			opt_dir = np.array(self.cfg.target) - self.cfg.init_state[:2]
			# perpendicular to target direction
			force_vec = np.array([opt_dir[1], -opt_dir[0]])
			deviation = ((obs[:2] - np.array(self.cfg.init_state[:2])) @ force_vec/np.linalg.norm(force_vec)) * force_vec

			action_pert += self.cfg.force_field_scale * deviation

		return action_pert
	
	def perturb_obs(self, obs):
		"""Perturb observation space."""
		return obs * (1. + self.cfg.obs_noise * torch.randn_like(obs))

	def set_init_state(self):
		phys = self.env.unwrapped.physics
		phys.set_state(np.array(self.cfg.init_state)) # fixed initial conditions

		return torch.tensor(self.cfg.init_state)

	def set_env(self):
		
		phys = self.env.unwrapped.physics
		phys.named.model.geom_size['target', 0] = self.cfg.target_size 
		phys.named.model.geom_margin['target'] = self.cfg.target_margin 
		
		# angle = self.cfg.target[0]
		# radius = self.cfg.target[1]
		# phys.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
		# phys.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

		phys.named.model.geom_pos['target', 'x'] = self.cfg.target[0]
		phys.named.model.geom_pos['target', 'y'] = self.cfg.target[1]

		self.env.unwrapped.task._control_cost = self.cfg.control_cost
		self.env.unwrapped.task._control_norm_thres = self.cfg.control_norm_thres

		self.env.unwrapped.task._target_reward = self.cfg.target_reward

		self.env.unwrapped.task._terminate_at_target = self.cfg.terminate_at_target

		if self.cfg.friction is not None:
			phys.named.model.geom_friction[:,0] = self.cfg.friction

		if self.cfg.terminate_at_target:
			self.env.unwrapped.task._end_reward = self.cfg.end_reward
			self.env.unwrapped.task._target_end_dist = self.cfg.target_end_dist
		

	def show_prediction(self, obs, action, action_eff=None, action_ctrl=None, action_total=None, future_obs=None):
		with torch.no_grad():
			pred, sig2 = self.agent.control_predict(obs, action)
		
		phys = self.env.unwrapped.physics
		phys.named.model.geom_pos['pred', 'x'] = pred[0]
		phys.named.model.geom_pos['pred', 'y'] = pred[1]
		phys.named.model.geom_size['pred', 0] = np.sqrt(sig2[0])
		
		if future_obs is not None:
			phys.named.model.geom_pos['pred_correct'][:2] = future_obs[:2]

		# print(phys.named.model.geom_pos['action_arrow'])
		phys.named.model.geom_pos['action_orig'][:2] = obs[:2] + action*self.cfg.action_arrow_scale

		if action_eff is not None:
			phys.named.model.geom_pos['action_eff'][:2] = obs[:2] + action_eff*self.cfg.action_arrow_scale

		if action_total is not None:
			phys.named.model.geom_pos['action_total'][:2] = obs[:2] + action_total*self.cfg.action_arrow_scale

		if action_ctrl is not None:
			phys.named.model.geom_pos['action_ctrl'][:2] = obs[:2] + action_ctrl*self.cfg.action_arrow_scale

		return pred, sig2

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True

		self.set_env()
		
		if self.cfg.perturb:
			self.perturb_factor = torch.zeros(self.env.action_space.shape[0])

		while self._step <= self.cfg.steps:

			# if self._step % 100 == 0:
			# 	print("Step:", self._step)

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if self._step > 0:
					ep_len = len(self._tds)
					# print("last episode length:", ep_len)
					# print("max episode length cfg:", self.cfg.episode_length)
					
					train_metrics.update({"episode_length": ep_len})

					# adaptive control
					if self.cfg.control:
						# for _ in range(self.cfg.ctrl_update_steps): # can't repeat online learning
						_train_metrics = self.agent.control_update(torch.cat(self._tds))
						train_metrics.update(_train_metrics)
						
						train_metrics.update({"perturb_factor": self.perturb_factor.numpy()})

					# Log episode
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')

					if ep_len > self.cfg.ctrl_horizon:
						# data = torch.cat(self._tds).detach().shape
						# print(self.buffer._storage._max_size_along_dim0(data))
						# print("maxsize:", self.buffer._storage.max_size)
						# print(data)
						data = torch.cat(self._tds).detach()
						# fill to episode length
						if ep_len < self.cfg.episode_length+1:
							fill_size = self.cfg.episode_length + 1 - ep_len
							empty_tds = [self.to_td(torch.zeros_like(data['obs'][0]), torch.zeros_like(data['action'][0]), torch.tensor(float('nan'))) for _ in range(fill_size)]
							data = torch.cat([data] + empty_tds)
						# add "done" flag
						data['valid'] = ~data['reward'].isnan()

						self._ep_idx = self.buffer.add(data)

					# Update agent
					if (self._step >= self.cfg.seed_steps) and self.cfg.train_agent:
						if self._step == self.cfg.seed_steps:
							num_updates = int(self.cfg.updates_per_step * self.cfg.seed_steps)
							print('Pretraining agent on seed data...')
							# num_updates = 10
						else:
							# num_updates = self.cfg.episode_length
							num_updates = int(self.cfg.updates_per_step * self.cfg.episode_length)
						for _ in range(num_updates):
							_train_metrics = self.agent.update(self.buffer)
						train_metrics.update(_train_metrics)

				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				obs = self.env.reset()

				if self.cfg.fixed_init_state:
					obs = self.set_init_state()

				self._tds = [self.to_td(obs)]

			# Collect experience
			if (self._step > self.cfg.seed_steps) or (not self.cfg.seed_random):
				action, action_orig, action_ctrl = self.agent.act(obs, t0=len(self._tds)==1, ctrl=self.cfg.control)
				# if self._step % 200 == 0:
					# print("Action:", action)
					# print("Action orig:", action_orig)
					# print("Action ctrl:", action_ctrl)
			else:
				action = self.env.rand_act()
				action_orig = action


			if self.cfg.perturb:
				effective_action = self.perturb(action, obs=obs)
			else:
				effective_action = action

			obs, reward, done, info = self.env.step(effective_action.detach())
			if self.cfg.perturb:
				obs = self.perturb_obs(obs)
			self._tds.append(self.to_td(obs, action_orig, reward)) # which action to use?

			if self.cfg.perturb and (self.cfg.perturb_scale > 0.) and (self._step >= self.cfg.seed_steps) and \
			(self._step % self.cfg.perturb_steps == 0):
				self.update_perturb()

			self._step += 1
	
		self.logger.finish(self.agent)
