import os
from torch.utils.data import Dataset
from PIL import Image
from util.datasets import build_transform
import torch
import pickle
import pandas as pd

class RAFDB(Dataset):
	def __init__(self, root, phase, args):

		self.root = os.path.join(root, "Image", "small")

		df = pd.read_csv(os.path.join(root, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
		NAME_COLUMN = 0
		LABEL_COLUMN = 1
		if phase == 'train':
			dataset = df[df[NAME_COLUMN].str.startswith('train')]
		else:
			dataset = df[df[NAME_COLUMN].str.startswith('test')]

		file_names = dataset.iloc[:, NAME_COLUMN].values
		label = (dataset.iloc[:, LABEL_COLUMN].values - 1)

		images = file_names.tolist()
		labels = label.tolist()

		self.images, self.labels = [], []

		num_class = args.nb_classes

		if 'train' == phase:
			self.isTrain = True
		else:
			self.isTrain = False

		for idx_item, item in enumerate(images):
			label = labels[idx_item]
			self.labels.append(label)
			self.images.append(item)

		self.cls_num_list = [0] * num_class

		for item in self.labels:
			self.cls_num_list[item] += 1

		print(len(self.images))
		print(self.cls_num_list)

		if phase == 'train':
			self.transform_train = build_transform(True, args)  ####  shoule be True
		else:
			self.transform_train = build_transform(False, args)

	def __len__(self, ):
		return len(self.images)

	def __getitem__(self, idx):
		img_name = self.images[idx]
		label = torch.tensor(self.labels[idx])
		image = Image.open(os.path.join(self.root, img_name))
		image = self.transform_train(image)

		if self.isTrain:
			return image, label, idx, img_name
		else:
			return image, label

def get_dataset(args):
	dataset_train = RAFDB(args.data_path, "train", args)
	dataset_val = RAFDB(args.data_path, "test", args)

	return dataset_train, dataset_val
