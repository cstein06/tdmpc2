import os
# os.environ['MUJOCO_GL'] = 'egl'
import sys
import warnings
# warnings.filterwarnings('ignore')
import torch

import hydra
import hydra.utils
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	# assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)

	if cfg.checkpoint_local:
		full_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_local)
		assert os.path.exists(full_path), f'Checkpoint {full_path} not found! Must be a valid filepath.'
		print("Loading checkpoint:", full_path)
		trainer.agent.load(full_path)
		# TODO
		print("This loading not implemented yet.")

	if cfg.checkpoint_wandb:
		print("Loading wandb artifact:", cfg.checkpoint_wandb)
		artifact = trainer.logger._wandb.run.use_artifact(cfg.checkpoint_wandb, type='model')
		artifact_dir = artifact.download()
		artifact_path = artifact_dir + f'/{cfg.checkpoint_filename}'
		print("Artifact path:", artifact_path)
		trainer.agent.load(artifact_path)
    
	try:
		trainer.train()
	except KeyboardInterrupt:
		print('Interrupted. Saving model and exiting...')
		trainer.logger.finish(trainer.agent)
		try:
			sys.exit(130)
		except SystemExit:
			os._exit(130)

	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
