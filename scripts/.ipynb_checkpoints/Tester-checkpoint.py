import math
import logging
import json

from tqdm import tqdm 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from sklearn.metrics import precision_recall_curve, auc
from lifelines.utils import concordance_index

logger = logging.getLogger(__name__)

class TesterConfig:
	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self,k,v)

class Tester:
	def __init__(self,model,testset, test_config):
		self.model = model
		self.testset = testset
		self.config = test_config

	def test(self):
		def reduce_fn(vals):
			return sum(vals)/len(vals)

		def test_loop_fn(test_dataloader, model,device):
			losses = []
			targets = []
			predictions = []
			model.eval()

			with torch.no_grad():
				for itr, (x,y) in enumerate(test_dataloader):
					x = x.to(device)
					y = y.to(device)

					pred, loss = model(x,y)
					loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn)
					losses.append(loss_reduced.item())

					target_itr = y.cpu().detach().numpy().tolist()
					predict_itr = pred.cpu().detach().numpy().tolist()

					targets.extend(target_itr)
					predictions.extend(predict_itr)

			return float(np.mean(losses)), targets, predictions

		test_sampler = DistributedSampler(
				self.testset,
				num_replicas=xm.xrt_world_size(),
				rank=xm.get_ordinal(),
				shuffle=False)

		test_dataloader = DataLoader(
				self.testset,
				batch_size=self.config.batch_size,
				sampler=test_sampler,
				num_workers=self.config.num_workers,
				drop_last=True)

		device = xm.xla_device()
		self.model.load_state_dict(torch.load(f'{self.config.ckpt_path}_{self.config.max_epoch}epoch_best_model.pt'))
		model = self.model.to(device)
		para_loader = pl.ParallelLoader(test_dataloader, [device]).per_device_loader(device)
		test_loss, T, P = test_loop_fn(para_loader, model, device)


		##Calculating CI and AUPR metrics:
		list_t = np.asarray(T).reshape(-1,).tolist()
		list_p = np.asarray(P).reshape(-1,).tolist()

		reference_binary_bindings = []
		predicted_binary_bindings = []

		for i in range(len(list_t)):
			if list_t[i]>=12.1:
				reference_binary_bindings.append(1)
			else:
				reference_binary_bindings.append(0)

		for i in range(len(list_p)):
			if list_p[i]>=12.1:
				predicted_binary_bindings.append(1)
			else:
				predicted_binary_bindings.append(0)


		percision, recall, _ = precision_recall_curve(reference_binary_bindings,predicted_binary_bindings) 
		test_AUPR = auc(recall, percision)

		test_ci = concordance_index(list_t, list_p)

		with open(f'{self.config.ckpt_path}_{self.config.max_epoch}epoch_results.json', 'w') as fp:
			json.dump({'CI':test_ci, 'MSE':test_loss, 'AUPR':test_AUPR}, fp)

		xm.master_print(f'CI:{test_ci} , MSE:{test_loss} , AUPR:{test_AUPR}')



