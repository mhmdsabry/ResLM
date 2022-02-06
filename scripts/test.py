import configparser
import argparse 

import torch
from torch.utils.data import TensorDataset
import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla
import torch_xla.core.xla_model as xm

from DTI_model import DTI_model, modelConfig
from Tester import Tester, TesterConfig
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

#dataset:
#Dataset:
kiba_dataset_path = config['kiba_dataset']['kiba_path']

device = 'cpu'

_, _, DTI_test, affinity_test = generate_DTI(kiba_dataset_path)
DTI_test = torch.from_numpy(DTI_test).to(device=device, dtype=torch.long)
affinity_test = torch.from_numpy(affinity_test).to(device=device, dtype=torch.long)

kiba_test = TensorDataset(DTI_test,affinity_test)

num_workers = int(config['training_config']['num_workers'])
batch_size = int(config['training_config']['eval_batch_size'])
ckpt_path = config['training_config']['ckpt_path']
max_epoch = int(config['training_config']['max_epoch'])
TPU = config['training_config']['TPU']

from DTI_model import DTI_model, modelConfig

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

test_config = TesterConfig(
							num_workers = num_workers,
							batch_size = batch_size,
							max_epoch = max_epoch,
							ckpt_path = ckpt_path
							)

tester = Tester(model, kiba_test, test_config)

def _map_fn(index):
	# For xla_spawn (TPUs)
	tester.test()

if __name__ == "__main__":
	if TPU:
		xmp.spawn(_map_fn, args=(), nprocs=num_workers,start_method='fork')
	else:
		tester.test()
