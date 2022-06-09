import os
from torch.utils.data import Dataset
from PIL import Image
from util.datasets import build_transform
import torch


class AffectNet(Dataset):
	def __init__(self, root, phase, args):

		self.root = os.path.join(root, phase)

		images = os.listdir(self.root)
		self.images, self.labels = [], []
		num_class = args.nb_classes

		for item in images:
			label = int(item[0])
			if label < num_class:
				self.labels.append(label)
				self.images.append(item)

		self.cls_num_list = [0] * num_class

		for item in self.labels:
			self.cls_num_list[item] += 1

		print(len(self.images))
		print(self.cls_num_list)

		self.pixel_std = 200
		self.aspect_ratio = 1.0

		if phase == 'training':
			self.transform_train = build_transform(False, args)
		else:
			self.transform_train = build_transform(False, args)

	def __len__(self, ):
		return len(self.images)

	def __getitem__(self, idx):
		img_name = self.images[idx]
		label = self.labels[idx]
		image = Image.open(os.path.join(self.root, img_name))
		image = self.transform_train(image)
		return image, label, idx,img_name


class ImbalancedDatasetSampler_SCN(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):

		# if indices is not provided,
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided,
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) \
			if num_samples is None else num_samples

		# distribution of classes in the dataset
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			# spdb.set_trace()
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1

		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
		           for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset.labels[idx]

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples


def get_dataset(args):
	dataset_train = AffectNet(args.data_path, "training", args)
	dataset_val = AffectNet(args.data_path, "validation", args)

	sampler = ImbalancedDatasetSampler_SCN(dataset_train)
	data_loader_train = torch.utils.data.DataLoader(
		dataset_train, sampler=sampler,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True,
	)

	data_loader_val = torch.utils.data.DataLoader(
		dataset_val, sampler=None,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)

	return data_loader_train, data_loader_val
