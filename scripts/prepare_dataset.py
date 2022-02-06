import pickle
import json
import numpy as np 
from collections import OrderedDict
import math
from tqdm import tqdm

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 63, "%": 64, ")": 65, "(": 66, "+": 67, "-": 68, 
			 ".": 69, "1": 70, "0": 71, "3": 72, "2": 73, "5": 74, 
			 "4": 75, "7": 76, "6": 77, "9": 78, "8": 79, "=": 80, 
			 "A": 81, "C": 82, "B": 83, "E": 84, "D": 85, "G": 86,
			 "F": 87, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

VOCAB_SIZE = 88
SMILEN = 320
SEQLEN = 1728

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]
	return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]
	return X #.tolist()

def parse_data(dataset_path): 
	fpath = dataset_path
	print("Read %s start" % fpath)

	ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
	proteins = json.load(open(fpath+"proteins.txt"), object_pairs_hook=OrderedDict)

	XD = []
	XT = []

	for d in ligands.keys():
		  XD.append(label_smiles(ligands[d], SMILEN, CHARCANSMISET))

	for t in proteins.keys():
		  XT.append(label_sequence(proteins[t], SEQLEN, CHARPROTSET))
 
	Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
	#Y = -(np.log10(Y/(math.pow(10,9))))

	return XD, XT, Y

def read_sets(dataset_path): ### fpath should be the dataset folder /kiba/ or /davis/
	fpath = dataset_path
	
	print("Reading %s start" % fpath)

	test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
	train_folds = json.load(open(fpath + "folds/train_fold_setting1.txt"))

	return  train_folds, test_fold

def prepare_interaction_pairs(XD, XT,  Y, rows, cols, train_folds, test_folds):
	DTI = np.zeros((len(rows), len(XD[0]) + len(XT[0])))
	affinity = np.zeros(len(rows))

	for pair_ind in tqdm(range(len(rows)), 'Start Pairing'):
		drug = XD[rows[pair_ind]]
		target = XT[cols[pair_ind]]
		DTI[pair_ind] = np.concatenate((drug, target))
		affinity[pair_ind] = Y[rows[pair_ind], cols[pair_ind]]
	
	
	DTI_train = DTI[train_folds[0]+train_folds[1]+train_folds[2]+train_folds[3]+train_folds[4]]
	affinity_train = affinity[train_folds[0]+train_folds[1]+train_folds[2]+train_folds[3]+train_folds[4]]
	DTI_test = DTI[test_folds]
	affinity_test = affinity[test_folds]

	return DTI_train, affinity_train, DTI_test, affinity_test



def generate_DTI(dataset_path):
  XD, XT, Y = parse_data(dataset_path)
  train_folds, test_fold = read_sets(dataset_path)
  XD = np.asarray(XD)
  XT = np.asarray(XT)
  Y = np.asarray(Y)
  label_row_inds, label_col_inds = np.where(np.isnan(Y)==False) 
  DTI_train, affinity_train, DTI_test, affinity_test = prepare_interaction_pairs(XD, XT, Y, label_row_inds, label_col_inds, train_folds, test_fold)
  return DTI_train, affinity_train, DTI_test, affinity_test
