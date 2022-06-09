import os
from torch.utils.data import Dataset
from util.datasets import build_transform
import torch
import pickle

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class fer2013_plus(Dataset):
	def __init__(self, root, phase, args):

		if "Train" in phase:
			self.isTrain = True
		else:
			self.isTrain = False

		self.root = os.path.join(root, phase)
		images = os.listdir(self.root)

		self.images, self.labels = [], []

		num_class = 8

		if 'FER2013Train' == phase:
			self.isTrain = True
			affectnet_partialY = pickle.load(open(args.labelset, "rb"))
			self.label_set = []

			for item in os.listdir(os.path.join(self.root)):
				label = int(item[0])
				self.labels.append(label)
				self.images.append(item)
				if item in affectnet_partialY:
					temp = [0] * num_class
					for label_ind in affectnet_partialY[item]:
						temp[label_ind] = 1
					self.label_set.append(temp)
				else:
					temp = [0] * num_class
					temp[label] = 1
					self.label_set.append(temp)
		else:
			self.isTrain = False
			for item in images:
				label = int(item[0])
				self.labels.append(label)
				self.images.append(item)

		self.cls_num_list = [0] * num_class

		for item in self.labels:
			self.cls_num_list[item] += 1

		print(len(self.images))
		print(self.cls_num_list)

		if phase == 'FER2013Train':
			self.transform_train = build_transform(True, args)
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
			return image, torch.tensor(self.label_set[idx]), label, idx
		else:
			return image, label


def get_dataset(args):
	dataset_train = fer2013_plus(args.data_path, "FER2013Train", args)
	dataset_val = fer2013_plus(args.data_path, "FER2013Test", args)

	return dataset_train, dataset_val, torch.tensor(dataset_train.label_set)
