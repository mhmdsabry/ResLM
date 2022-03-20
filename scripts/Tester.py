import math
import logging
import json

from tqdm import tqdm 
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
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
         
	def measure_LM_contribution(self, model, x, targets):
		b, t = x.size()
		x = model.embed(x)
		x = model.query_block_LN(x)
		x = x + model.query_block(x).transpose(1,2)
		x = model.head_LN(x)
		y = model.head(x)
		y = y.mean(1).reshape((b,))
		loss = F.mse_loss(y, targets.float())
		return y, loss

	def test(self):
		def reduce_fn(vals):
			return sum(vals)/len(vals)

		def test_loop_fn(test_dataloader, model,device):
			losses = []
			no_lm_losses = []
			targets = []
			predictions = []
			no_lm_predictions = []
			model.eval()

			with torch.no_grad():
				for itr, (x,y) in enumerate(test_dataloader):
					x = x.to(device)
					y = y.to(device)

					target_itr = y.cpu().detach().numpy().tolist()
					targets.extend(target_itr)

					pred, loss = model(x,y)
					loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn)
					losses.append(loss_reduced.item())                    
					predict_itr = pred.cpu().detach().numpy().tolist()
					predictions.extend(predict_itr)
                    
					if self.config.measure_LM_contribution:
						no_lm_pred, no_lm_loss = self.measure_LM_contribution(model, x, y)
						no_lm_loss_reduced = xm.mesh_reduce('loss_reduce',no_lm_loss,reduce_fn)
						no_lm_losses.append(no_lm_loss_reduced.item())
						no_lm_predict_itr = no_lm_pred.cpu().detach().numpy().tolist()
						no_lm_predictions.extend(no_lm_predict_itr)

			if self.config.measure_LM_contribution:
				return float(np.mean(losses)), predictions, targets, float(np.mean(no_lm_losses)), no_lm_predictions
			else:
				return float(np.mean(losses)), predictions, targets

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
        
		if self.config.measure_LM_contribution:       
			test_loss, P, T, no_lm_test_loss, no_lm_P = test_loop_fn(para_loader, model, device)
			##Calculating CI and AUPR metrics:
			list_t = np.asarray(T).reshape(-1,).tolist()
			list_p = np.asarray(P).reshape(-1,).tolist()
			no_lm_list_p = np.asarray(no_lm_P).reshape(-1,).tolist()

			reference_binary_bindings = []
			predicted_binary_bindings = []
			no_lm_predicted_binary_bindings = []

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
                    
			for i in range(len(no_lm_list_p)):
				if no_lm_list_p[i]>=12.1:
					no_lm_predicted_binary_bindings.append(1)
				else:
					no_lm_predicted_binary_bindings.append(0)

			percision, recall, _ = precision_recall_curve(reference_binary_bindings, predicted_binary_bindings)
			no_lm_percision, no_lm_recall, _ = precision_recall_curve(reference_binary_bindings, no_lm_predicted_binary_bindings)
			test_AUPR = auc(recall, percision)
			no_lm_test_AUPR = auc(no_lm_recall, no_lm_percision)
            
			test_ci = concordance_index(list_t, list_p)          
			no_lm_test_ci = concordance_index(list_t, no_lm_list_p)
            
			with open(f'{self.config.ckpt_path}_{self.config.max_epoch}epoch_results.json', 'w') as fp:
				json.dump({'CI':test_ci, 'MSE':test_loss, 'AUPR':test_AUPR,
							'no_lm_CI':no_lm_test_ci, 'no_lm_MSE':no_lm_test_loss, 'no_lm_AUPR':no_lm_test_AUPR}, fp)
			xm.master_print(f'  LM: CI:{test_ci} , MSE:{test_loss} , AUPR:{test_AUPR}\n NoLM: CI:{no_lm_test_ci} , MSE:{no_lm_test_loss} , AUPR:{no_lm_test_AUPR}')
            
		else:
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



