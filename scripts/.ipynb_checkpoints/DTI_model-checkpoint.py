import math
import logging

import numpy as np 

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Config
from transformers import BertModel, T5Model

import torch
import torch.nn as nn
from torch.nn import functional as F 

logger = logging.getLogger(__name__)

class modelConfig:

	embed_pdrop = 0.1
	query_block_pdrop = 0.1
	LM_block_pdrop = 0.1

	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self,k,v)

class query_block(nn.Module):
	def __init__(self,config):
		super().__init__()

		in_dim = config.query_block_conv_in_dim
		out_dim = config.query_block_conv_out_dim
		kernel_size = config.query_block_conv_kernel_size
		stride = config.query_block_conv_stride
		padding = config.query_block_conv_padding
		input_size = config.query_block_recurrent_input_size
		hidden_size = config.query_block_recurrent_hidden_size 
		num_layers  = config.query_block_recurrent_num_layers
		recurrent_in_dim = config.query_block_recurrent_read_dim
		recurrent_out_dim = config.query_block_recurrent_write_dim
		self.Is_Recurrent = config.Is_Recurrent
        
		if self.Is_Recurrent:
			self.in_layer = nn.Linear(recurrent_in_dim, recurrent_out_dim)
			self.rnn = nn.RNN(input_size, hidden_size, num_layers)
			self.out_layer = nn.Linear(recurrent_out_dim, recurrent_in_dim)
		else:
			self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
			self.relu = nn.ReLU()
		self.dropout = nn.Dropout(config.query_block_pdrop)

	def forward(self,x):
		if self.Is_Recurrent:
			x = x.transpose(1,2)
			x = self.in_layer(x).transpose(1,2)
			x_mean = torch.mean(x, 1, True).transpose(0,1)
			x_var = torch.var(x, 1, unbiased=False).unsqueeze(dim=0)
			h_0 = torch.stack([x_mean,x_var], dim=0).squeeze(dim=1)
			x = x.transpose(0,1)
			x, _ = self.rnn(x,h_0)
			x = self.out_layer(x.permute((1,2,0)))
		else:
			x = self.relu(self.conv1d(x))
		x = self.dropout(x)

		return x 


class LM_block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		in_dim = config.LM_block_read_dim
		out_dim = config.LM_block_write_dim

		#We will read lineary from the residual stream and write lineary to it 
		self.in_layer = nn.Linear(in_dim, out_dim)

		if config.LM == "GPT2":
			self.groupnorm = nn.GroupNorm(config.groupnorm, 1024) #the motivations for groupnorm is to group features crrosbonding to drug-target groups
			self.lm = GPT2Model.from_pretrained('gpt2')

			# freeze all parameters except the layernorm and positional embeddings
			for name, param in self.lm.named_parameters():
				if 'ln' in name or 'wpe' in name:
					param.requires_grad = True
				else:
					param.requires_grad = False

		elif config.LM == "T5" or "mBert" or "Bert":
			self.groupnorm = nn.GroupNorm(config.groupnorm, 512)

			if config.LM == "T5":
				self.lm = T5Model.from_pretrained('t5-base')

				# freeze all parameters except the layernorm and positional embeddings
				for name, param in self.lm.named_parameters():
					if 'position_embeddings' in name or 'layer_norm' in name:
						param.requires_grad = True 
					else:
						param.requires_grad = False
		
			elif config.LM == "mBert":
				self.lm = BertModel.from_pretrained('bert-base-multilingual-uncased')

				# freeze all parameters except the layernorm and positional embeddings
				for name, param in self.lm.named_parameters():
					if 'position_embeddings' in name or 'LayerNorm' in name:
						param.requires_grad = True 
					else:
						param.requires_grad = False

			elif config.LM == "Bert":
				self.lm  = BertModel.from_pretrained('bert-base-uncased')

				# freeze all parameters except the layernorm and positional embeddings
				for name, param in self.lm.named_parameters():
					if 'position_embeddings' in name or 'LayerNorm' in name:
						param.requires_grad = True 
					else:
						param.requires_grad = False

		self.out_layer = nn.Linear(out_dim,in_dim) #dims are opposite in_layer becuase we want the same dims recieved from residual stream to be out
		self.dropout = nn.Dropout(config.LM_block_pdrop)

	def forward(self, x):
		x = x.transpose(1,2)
		x = self.in_layer(x).transpose(1,2)
		x = self.groupnorm(x)
		if self.config.LM == "T5":
			x = self.lm(inputs_embeds=x, decoder_inputs_embeds=x, output_attentions=True).last_hidden_state[:,:]
		else:
			x = self.lm(inputs_embeds=x, output_attentions=True).last_hidden_state[:,:]
		x = self.out_layer(x.transpose(1,2))
		x = self.dropout(x)
		x = x.transpose(1,2)

		return x 



class DTI_model(nn.Module):
	def __init__(self,config):
		super().__init__()

		if config.LM == "GPT2" or "T5" or "mBert" or "Bert":
			#Bert n_embed=768, so we will keep that through our architecture
			self.embed = nn.Embedding(config.vocab_size, 768)
			self.embed_drop = nn.Dropout(config.embed_pdrop)

			self.query_block_LN = nn.LayerNorm(768)
			self.query_block = query_block(config)

			self.LM_block_LN = nn.LayerNorm(768)
			self.LM_block = LM_block(config)

			self.head_LN = nn.LayerNorm(768)
			self.head = nn.Linear(768, 1, bias=False)

			self.apply(self._init_weights)
		else:
			print("Please choose LM in model config to be either GPT2, T5, mBert, or Bert")

		logger.info("number of all parameters: %e", sum(p.numel() for p in self.parameters()))
		logger.info("number of  trainable parameters: %e", sum(p.numel() for p in self.parameters() if p.requires_grad==True))

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def configure_optimizers(self, train_config):

		"""
		Here we will seperate the parameters into two subsets, one will experience weight decay
		and the other won't (layernorm, embedding weights and biases)
		we then return pytorch optimizer

		"""
		decay = set()
		no_decay = set()

		blacklist_modules = (nn.LayerNorm, nn.GroupNorm, nn.Embedding)
		whitelist_modules = (nn.Linear, nn.Conv1d)

		for mn, m in self.named_modules():
			for pn, p in m.named_parameters():
				fpn = "%s.%s" %(mn,pn) if mn else pn 

				if pn.endswith('bias'):
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m,whitelist_modules):
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m,blacklist_modules):
					no_decay.add(fpn)

		params_dict = {pn:p for pn, p in self.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay

		assert len(inter_params) == 0, "Parameters %s are in decay/no_decay sets!"%(str(union_params))

		optim_group = [
			{"params":[params_dict[pn] for pn in sorted(list(decay))], "weight_decay":train_config.weight_decay},
			{"params":[params_dict[pn] for pn in sorted(list(no_decay))], "weight_decay":0.0}
		]
		optimizer = torch.optim.AdamW(optim_group, lr=train_config.learning_rate, betas=(train_config.betas_1,train_config.betas_2))
		return optimizer

	def forward(self, x, targets):
		b, t = x.size()

		x = self.embed(x)

		x = self.query_block_LN(x)
		x = x + self.query_block(x).transpose(1,2)
        
		x = self.LM_block_LN(x)
		x = x + self.LM_block(x)

		x = self.head_LN(x)
		y = self.head(x)
		y = y.mean(1).reshape((b,))
		loss = F.mse_loss(y, targets.float())

		return y, loss 















