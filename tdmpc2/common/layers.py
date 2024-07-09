import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble
# from torch.func import stack_module_state

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		modules = nn.ModuleList(modules)

		# self.modules_ = modules

		# fn, params = stack_module_state(modules)
		# def wrapper(params, buffers, data):
		# 	return torch.func.functional_call(modules[0], (params, buffers), data)
		# self.vmap = torch.vmap(params, in_dims=(0, 0, None), randomness='different', **kwargs)

		fn, params, _ = combine_state_for_ensemble(modules)
		self.vmap = torch.func.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)

		self.params = nn.ParameterList([nn.Parameter(p) for p in params])
		self._repr = str(modules)

	# def modules(self):
	# 	return self.modules_
	# 	# return self.vmap
	# 	# return self.vmap.__wrapped__.stateless_model

	def forward(self, *args, **kwargs):
		return self.vmap([p for p in self.params], (), *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., activ=None, **kwargs):
		super().__init__(*args, **kwargs)
		if activ is None:
			activ = nn.Mish(inplace=True)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = activ
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))
	
	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		assert k == 'state'
		out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, out_activ=SimNorm(cfg) if cfg.use_simnorm else None)
	return nn.ModuleDict(out)


def mlp(in_dim, mlp_dims, out_dim, hidden_activ=None, out_activ=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0), activ=hidden_activ))
	mlp.append(NormedLinear(dims[-2], dims[-1], activ=out_activ) if out_activ else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)
