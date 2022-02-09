import math
import logging
import json

from tqdm import tqdm 
import numpy as np 

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# +
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import get_linear_schedule_with_warmup
# -

logger = logging.getLogger(__name__)


class TrainerConfig:
	max_epoch = 10
	batch_size = 64
	warmup_tokens = 234e2
	num_workers = 4
	ckpt_path = None

	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self,k,v)

class Trainer:
	def __init__(self, model, trainset, evalset, train_config):
		self.model = model
		self.trainset = trainset
		self.evalset = evalset
		self.config = train_config

	def saving_checkpoints(self,timeline):
		model = self.model.module if hasattr(self.model, "module") else self.model
		logger.info("Saving at %s", self.config.ckpt_path)
		xm.save(model.state_dict(), f"{self.config.ckpt_path}{timeline}.pt")

	def train(self):
		train_state = {
							"epoch":[],
							"train_loss": [],
							"eval_loss":[],
							"best_loss": float('inf')
							}
        
		def reduce_fn(vals):
			return sum(vals)/len(vals)
        
		def train_loop_fn(train_dataloader,model,optimizer,device,scheduler=None):
			losses = []
			model.train()
			#pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) 
			for itr, (x, y) in enumerate(train_dataloader):
				x = x.to(device)
				y = y.to(device)

				optimizer.zero_grad()
				pred, loss = model(x,y)
				loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn)
				losses.append(loss_reduced.item())

				if itr % 50 ==0:
					xm.master_print(f'Itr={itr}, loss={loss_reduced}')

				loss.backward()
				xm.optimizer_step(optimizer)
				if scheduler is not None:
					scheduler.step()

			return float(np.mean(losses))

		def eval_loop_fn(eval_dataloader, model,device):
			losses = []
			model.eval()
			for itr, (x,y) in enumerate(eval_dataloader):
				x = x.to(device)
				y = y.to(device)

				pred, loss = model(x,y)
				loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn)
				losses.append(loss_reduced.item())

			return float(np.mean(losses))


		device = xm.xla_device()
		model = self.model.to(device)
		optimizer = model.configure_optimizers(self.config)
		num_train_steps = int(len(self.trainset) / self.config.train_batch_size / xm.xrt_world_size() * self.config.max_epoch)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=self.config.warmup_steps*num_train_steps,
			num_training_steps=num_train_steps
			)
        
		train_sampler = DistributedSampler(
				self.trainset,
				num_replicas=xm.xrt_world_size(),
				rank=xm.get_ordinal(),
				shuffle=False)
		eval_sampler = DistributedSampler(
				self.evalset,
				num_replicas=xm.xrt_world_size(),
				rank=xm.get_ordinal(),
				shuffle=False)

		train_dataloader = DataLoader(
				self.trainset,
				batch_size= self.config.train_batch_size,
				sampler= train_sampler,
				num_workers= self.config.num_workers,
				drop_last=True)
		eval_dataloader = DataLoader(
				self.evalset,
				batch_size=self.config.eval_batch_size,
				sampler=eval_sampler,
				num_workers=self.config.num_workers,
				drop_last=True)


		best_loss = float('inf')
		for epoch in range(self.config.max_epoch):
			xm.master_print(f'epoch={epoch+1}/ {self.config.max_epoch}')
			train_state['epoch'].append(epoch+1)

			para_loader = pl.ParallelLoader(train_dataloader, [device]).per_device_loader(device)
			train_loss = train_loop_fn(para_loader, model, optimizer, device, scheduler=scheduler)
			train_state['train_loss'].append(train_loss)

			para_loader = pl.ParallelLoader(eval_dataloader, [device]).per_device_loader(device)
			eval_loss = eval_loop_fn(para_loader, model, device)
			train_state['eval_loss'].append(eval_loss)

			good_model = eval_loss < best_loss             
			if self.config.ckpt_path is not None and good_model:
				best_loss = eval_loss
				train_state['best_loss'] = best_loss
				self.saving_checkpoints(f"_{self.config.max_epoch}epoch_best_model")

		with open(f'{self.config.ckpt_path}_{self.config.max_epoch}epoch_train_state.json', 'w') as fp:
			json.dump(train_state, fp)
		self.saving_checkpoints(f"_{self.config.max_epoch}epoch_last_model")









