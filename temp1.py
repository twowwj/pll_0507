
import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from timm.utils import accuracy
# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import pickle
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_vit

from engine_finetune import train_one_epoch, evaluate
from torch.nn import DataParallel
from util.mlogger import create_logger

""" no augmentation of training dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_args_parser():
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'
	parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
	# append
	parser.add_argument('--dataset', default='rafdb', type=str)
	parser.add_argument('--data_path', default='/data/wjwang/fer_dataset/raf_0806', type=str, help='dataset path')
	parser.add_argument('--output_dir', default='/data/wjwang/pll_0507/log/', type=str)
	parser.add_argument('--nb_classes', default=7, type=int,
	                    help='number of the classification types')
	parser.add_argument('--ckpt_dir', default='/data/wjwang/pll_0507/pretrain-MF/ckpt', type=str)
	parser.add_argument('--labelset', default='/data0/wwang/pll_0507/labelset_rafdb/113_54.pkl', type=str)
	#
	parser.add_argument('--batch_size', default=1, type=int,
	                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
	parser.add_argument('--epochs', default=25, type=int)
	parser.add_argument('--accum_iter', default=1, type=int,
	                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

	# Model parameters
	parser.add_argument('--model', default='vit_base_affectnet', type=str, metavar='MODEL',
	                    help='Name of model to train')

	parser.add_argument('--input_size', default=224, type=int,
	                    help='images input size')

	parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
	                    help='Drop path rate (default: 0.1)')

	# Optimizer parameters
	parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
	                    help='Clip gradient norm (default: None, no clipping)')
	parser.add_argument('--weight_decay', type=float, default=0.05,
	                    help='weight decay (default: 0.05)')

	parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
	                    help='learning rate (absolute lr)')
	parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
	                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
	parser.add_argument('--layer_decay', type=float, default=0.75,
	                    help='layer-wise lr decay from ELECTRA/BEiT')

	parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
	                    help='lower lr bound for cyclic schedulers that hit 0')

	parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
	                    help='epochs to warmup LR')

	# Augmentation parameters
	parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
	                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
	parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
	                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
	parser.add_argument('--smoothing', type=float, default=0.1,
	                    help='Label smoothing (default: 0.1)')

	# * Random Erase params
	parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
	                    help='Random erase prob (default: 0.25)')
	parser.add_argument('--remode', type=str, default='pixel',
	                    help='Random erase mode (default: "pixel")')
	parser.add_argument('--recount', type=int, default=1,
	                    help='Random erase count (default: 1)')
	parser.add_argument('--resplit', action='store_true', default=False,
	                    help='Do not random erase first (clean) augmentation split')


	# * Finetuning params
	parser.add_argument('--finetune', default='',
	                    help='finetune from checkpoint')
	parser.add_argument('--resume', default='',
	                    help='finetune from checkpoint')
	parser.add_argument('--global_pool', action='store_true')
	parser.set_defaults(global_pool=True)
	parser.add_argument('--cls_token', action='store_false', dest='global_pool',
	                    help='Use class token instead of global pool for classification')

	# Dataset parameters
	parser.add_argument('--device', default='cuda',
	                    help='device to use for training / testing')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
	                    help='start epoch')
	parser.add_argument('--dist_eval', action='store_true', default=False,
	                    help='Enabling distributed evaluation (recommended during training for faster monitor')
	parser.add_argument('--num_workers', default=10, type=int)
	parser.add_argument('--pin_mem', action='store_true',
	                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
	parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
	parser.set_defaults(pin_mem=True)
	return parser



@torch.no_grad()
def evaluate(data_loader, model, device, logger):
	criterion = torch.nn.CrossEntropyLoss()

	metric_logger = misc.MetricLogger(delimiter="  ")
	header = 'Test:'

	# switch to evaluation mode
	model.eval()
	affectnet_partialY = pickle.load(open(args.labelset, "rb"))
	label_new_set = {}
	count =0
	for batch in metric_logger.log_every(data_loader, 10, header, logger=logger):
		images = batch[0]
		target = batch[1]
		image_name = batch[3][0]
		images = images.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)
		# compute output
		with torch.cuda.amp.autocast():
			output = model(images)
			loss = criterion(output, target)
			acc1, acc2, acc3, acc5 = accuracy(output, target, topk=(1, 2, 3, 5))
			if acc1 == 0:
				index_top3 = output.topk(3)[1].cpu().detach().numpy().tolist()
				label_new_set[image_name] = index_top3[0]
				count+=1


		batch_size = images.shape[0]
		metric_logger.update(loss=loss.item())
		metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
		metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
		metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
		metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

	# gather the stats from all processes
	with open('/data0/wwang/pll_0507/labelset_rafdb/new_labelset—top3.pkl', 'wb') as f:
		pickle.dump(label_new_set, f)
	metric_logger.synchronize_between_processes()


	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
	# a = pickle.load(open('/data0/wwang/pll_0507/labelset_rafdb/22_182.pkl', "rb"))
	device = torch.device(args.device)

	if args.dataset == 'rafdb':
		args.nb_classes = 7
		from datasets.rafdb_base import get_dataset
	elif args.dataset == 'ferplus':
		args.nb_classes = 8
		from datasets.ferplus_base import get_dataset
	elif args.dataset == 'affectnet7':
		from datasets.affetnet_base import get_dataset
		args.nb_classes = 7
	elif args.dataset == 'affectnet8':
		from datasets.affetnet_base import get_dataset
		args.nb_classes = 8

	# dataset_train, dataset_val = get_dataset(args)
	# sampler_train = torch.utils.data.RandomSampler(dataset_train)
	# sampler_val = torch.utils.data.SequentialSampler(dataset_val)
	#
	# data_loader_train = torch.utils.data.DataLoader(
	# 	dataset_train, sampler=sampler_train,
	# 	batch_size=args.batch_size,
	# 	num_workers=args.num_workers,
	# 	pin_memory=args.pin_mem,
	# 	drop_last=True,
	# )
	# data_loader_val = torch.utils.data.DataLoader(
	# 	dataset_val, sampler=sampler_val,
	# 	batch_size=args.batch_size,
	# 	num_workers=args.num_workers,
	# 	pin_memory=args.pin_mem,
	# 	drop_last=False
	# )

	data_loader_train, data_loader_val = get_dataset(args)

	save_path = os.path.join(args.output_dir)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	logger = create_logger(save_path)

	model = models_vit.__dict__[args.model](
		num_classes=args.nb_classes,
		drop_path_rate=args.drop_path,
		global_pool=args.global_pool,
	)


	model.to(device)
	model_without_ddp = model
	param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
										no_weight_decay_list=model_without_ddp.no_weight_decay(),
										layer_decay=args.layer_decay
										)
	loss_scaler = NativeScaler()
	optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
					loss_scaler=loss_scaler)


	# test_stats = evaluate(data_loader_val, model, device, logger)
	test_stats = evaluate(data_loader_train, model, device, logger)
	print(test_stats)
if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	main(args)
