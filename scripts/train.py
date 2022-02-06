import configparser
import argparse 

import torch
from torch.utils.data import TensorDataset
import torch_xla.distributed.xla_multiprocessing as xmp

from DTI_model import DTI_model, modelConfig
from Trainer import Trainer, TrainerConfig
from utils import *
from prepare_dataset import *

SEED = 3056
seed_everything(SEED)

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("-c","--config",dest="filename", help="Pass a training config file",metavar="FILE")
args = parser.parse_args()
config.read(args.filename)


#hyperparameters:
#Dataset:
kiba_dataset_path = config['kiba_dataset']['kiba_path']
kiba_train_len = int(config['kiba_dataset']['kiba_train_len'])
kiba_val_len = int(config['kiba_dataset']['kiba_val_len'])

#prepare dataset
#do this step on cpu
device = 'cpu'

DTI_train, affinity_train, DTI_test, affinity_test = generate_DTI(kiba_dataset_path)
DTI_train = torch.from_numpy(DTI_train).to(device=device, dtype=torch.long)
affinity_train = torch.from_numpy(affinity_train).to(device=device, dtype=torch.long)
DTI_test = torch.from_numpy(DTI_test).to(device=device, dtype=torch.long)
affinity_test = torch.from_numpy(affinity_test).to(device=device, dtype=torch.long)

trainset_tokens = len(DTI_train) * 2048

kiba_train = TensorDataset(DTI_train[:kiba_train_len],affinity_train[:kiba_train_len])
kiba_eval = TensorDataset(DTI_train[kiba_train_len:kiba_train_len+kiba_val_len],affinity_train[kiba_train_len:kiba_train_len+kiba_val_len])

#model
query_block_conv_in_dim = int(config['model_config']['query_block_conv_in_dim'])
query_block_conv_out_dim = int(config['model_config']['query_block_conv_out_dim'])
query_block_conv_stride = int(config['model_config']['query_block_conv_stride'])
query_block_conv_padding = int(config['model_config']['query_block_conv_padding'])
query_block_conv_kernel_size = int(config['model_config']['query_block_conv_kernel_size'])
LM = config['model_config']['LM']
groupnorm = int(config['model_config']['groupnorm'])
LM_block_read_dim = int(config['model_config']['LM_block_read_dim'])
LM_block_write_dim = int(config['model_config']['LM_block_write_dim'])

#Trainer
num_workers = int(config['training_config']['num_workers'])
max_epoch = int(config['training_config']['max_epoch'])
train_batch_size = int(config['training_config']['train_batch_size'])
eval_batch_size = int(config['training_config']['eval_batch_size'])
learning_rate = float(config['training_config']['learning_rate']) * num_workers
warmup_tokens = float(config['training_config']['warmup_tokens']) * trainset_tokens 
final_tokens = int(config['training_config']['final_tokens']) * trainset_tokens // num_workers
lr_decay = config['training_config']['lr_decay']
ckpt_path = config['training_config']['ckpt_path']
intermediate_tokens_saves = float(config['training_config']['intermediate_tokens_saves']) * trainset_tokens
weight_decay = float(config['training_config']['weight_decay'])
betas_1= float(config['training_config']['betas_1'])
betas_2= float(config['training_config']['betas_2'])
TPU = config['training_config']['TPU']

#prepare model
model_config = modelConfig(
							query_block_conv_in_dim = query_block_conv_in_dim,
							query_block_conv_out_dim = query_block_conv_out_dim,
							query_block_conv_stride = query_block_conv_stride,
							query_block_conv_padding = query_block_conv_padding,
							query_block_conv_kernel_size = query_block_conv_kernel_size,
							LM = LM,
							groupnorm = groupnorm,
							LM_block_read_dim = LM_block_read_dim,
							LM_block_write_dim = LM_block_write_dim,
							vocab_size = VOCAB_SIZE #set at prepare_dataset.py
							)

model = DTI_model(model_config)

#prepare trainer
train_config = TrainerConfig(
							max_epoch = max_epoch,
							train_batch_size = train_batch_size,
							eval_batch_size = eval_batch_size,
							learning_rate = learning_rate,
							warmup_tokens = warmup_tokens,
							final_tokens = final_tokens,
							lr_decay = lr_decay,
							num_workers = num_workers,
							ckpt_path = ckpt_path,
							intermediate_tokens_saves = intermediate_tokens_saves,
							weight_decay = weight_decay,
							betas_1 = betas_1,
							betas_2 = betas_2,
							TPU = TPU,

							)

trainer = Trainer(model, kiba_train, kiba_eval, train_config)

def _map_fn(index):
	# For xla_spawn (TPUs)
	trainer.train()

if __name__ == "__main__":
	if TPU:
		xmp.spawn(_map_fn, args=(), nprocs=num_workers,start_method='fork')
	else:
		trainer.train()















