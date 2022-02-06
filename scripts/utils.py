import math 
import json

import random, os
import numpy as np 
import torch

import transformers

import matplotlib.pyplot as plt


def seed_everything(seed: int):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	transformers.set_seed(seed)
	torch.manual_seed(seed)

def plot_learning_curve(state, saving_path=None):
	state = json.load(open(state))

	train_loss = state.get('train_loss')
	val_loss = state.get('eval_loss')
	epoch = state.get('epoch')

	fig = plt.figure()
	ax = plt.axes()
	ax.plot(epoch, train_loss,label='Train Loss')
	ax.plot(epoch, val_loss, label='Eval Loss')
	plt.xlabel("Epoch")
	plt.ylabel("Loss");
	plt.legend()

	if saving_path is not None:
		plt.savefig(f'{saving_path}_learning_curve.png')

	plt.show()


def plot_epoch_test_loss(epochs_model_paths, saving_path=None):
	epochs_model_losses = get_epoch_model_loss(epochs_model_paths)
	epoch = [i+1 for i in range(len(epochs_model_losses))]

	fig = plt.figure()
	ax = plt.axes()

	ax.plot(epoch, epochs_model_losses[0], label='GPT2')
	ax.plot(epoch, epochs_model_losses[1], label='T5')
	ax.plot(epoch, epochs_model_losses[2], label='Bert')
	ax.plot(epoch, epochs_model_losses[3], label='mBert')

	plt.xlabel("Epoch")
	plt.ylabel("Loss");
	plt.legend()

	if saving_path is not None:
		plt.savefig(f'{saving_path}_epoch.png')

	plt.show()


def get_epoch_model_loss(epochs_model_paths):
	pass












