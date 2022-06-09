# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

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

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_vit

from engine_finetune_rc import train_one_epoch, evaluate
from torch.nn import DataParallel
from util.mlogger import create_logger

""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] ='1'
def rc_loss_smooth(outputs, confidence, index, num_classes=8, lb_smooth=0.1):
	logsm_outputs = F.log_softmax(outputs, dim=1)
	label = confidence[index, :]
	temp2 = torch.where(label == 0, torch.zeros(1).cuda(), torch.ones(1).cuda()).sum(dim=1)
	lb_neg = lb_smooth / (num_classes - temp2)
	for i in range(outputs.shape[0]):
		label[i, label[i, :] > 0] -= lb_smooth / temp2[i]
		label[i, label[i, :] == 0] = lb_neg[i]

	final_outputs = logsm_outputs * label

	average_loss = -((final_outputs).sum(dim=1)).mean()
	return average_loss

def get_args_parser():
	# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
	# append
	parser.add_argument('--dataset', default='rafdb', type=str)
	parser.add_argument('--data_path', default='/data0/wwang/pll_0507/data/raf_0806', type=str, help='dataset path')
	parser.add_argument('--output_dir', default='/data0/wwang/pll_0507/', type=str)
	parser.add_argument('--exp', default='default', type=str)
	parser.add_argument('--nb_classes', default=7, type=int,
	                    help='number of the classification types')
	parser.add_argument('--ckpt_dir', default='/data/wjwang/pll_0507/pretrain-MF/ckpt', type=str)
	#
	parser.add_argument('--batch_size', default=4, type=int,
	                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
	parser.add_argument('--epochs', default=25, type=int)
	parser.add_argument('--accum_iter', default=1, type=int,
	                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
	parser.add_argument('--labelset', default='', type=str)
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

	# * Mixup params
	parser.add_argument('--mixup', type=float, default=0,
	                    help='mixup alpha, mixup enabled if > 0.')
	parser.add_argument('--cutmix', type=float, default=0,
	                    help='cutmix alpha, cutmix enabled if > 0.')
	parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
	                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
	parser.add_argument('--mixup_prob', type=float, default=1.0,
	                    help='Probability of performing mixup or cutmix when either/both is enabled')
	parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
	                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
	parser.add_argument('--mixup_mode', type=str, default='batch',
	                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

	# * Finetuning params
	parser.add_argument('--finetune', default='/data0/wwang/pll_0507/labelset_rafdb/22_182.pkl',
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



def save_ckpt(args, model_dict, epoch):
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	ckpt_name = os.path.join(args.ckpt_dir, "epoch_{}.pth".format(epoch))
	print("saving checkpoint into {}".format(ckpt_name))
	outdict = {}
	if isinstance(model_dict, torch.nn.DataParallel):
		model_dict = model_dict.module
	torch.save(model_dict, ckpt_name)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main(args):
	# ckpt = torch.load('/data/wjwang/pll_0507/pretrain-MF/ckpt/epoch_0.pth')

	setup_seed(args.seed)
	device = torch.device(args.device)

	if args.dataset == 'rafdb':
		args.nb_classes = 7
		from datasets.rafdb_pll import get_dataset
	elif args.dataset == 'ferplus':
		args.nb_classes = 8
		from datasets.ferplus_pll import get_dataset

	dataset_train, dataset_val, train_givenY = get_dataset(args)
	tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
	confidence = train_givenY.float() / tempY
	confidence = confidence.to(device)


	sampler_train = torch.utils.data.RandomSampler(dataset_train)
	sampler_val = torch.utils.data.SequentialSampler(dataset_val)

	data_loader_train = torch.utils.data.DataLoader(
		dataset_train, sampler=sampler_train,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True,
	)

	data_loader_val = torch.utils.data.DataLoader(
		dataset_val, sampler=sampler_val,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)


	save_path = os.path.join(args.output_dir, args.exp)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	logger = create_logger(save_path)
	logger.info("{}".format(args).replace(', ', ',\n'))

	mixup_fn = None
	mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
	if mixup_active:
		print("Mixup is activated!")
		mixup_fn = Mixup(
			mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
			prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
			label_smoothing=args.smoothing, num_classes=args.nb_classes)

	model = models_vit.__dict__[args.model](
		num_classes=args.nb_classes,
		drop_path_rate=args.drop_path,
		global_pool=args.global_pool,
	)

	if args.finetune:
		checkpoint = torch.load(args.finetune, map_location='cpu')

		logger.info("Load pre-trained checkpoint from: %s" % args.finetune)
		checkpoint_model = checkpoint['model']
		state_dict = model.state_dict()
		for k in ['head.weight', 'head.bias']:
			if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
				logger.info(f"Removing key {k} from pretrained checkpoint")
				del checkpoint_model[k]

		# interpolate position embedding
		interpolate_pos_embed(model, checkpoint_model)

		# load pre-trained model
		msg = model.load_state_dict(checkpoint_model, strict=False)
		logger.info(msg)

		if args.global_pool:
			assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
		else:
			assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

		# manually initialize fc layer
		trunc_normal_(model.head.weight, std=2e-5)

	model.to(device)

	model_without_ddp = model
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	logger.info("Model = %s" % str(model_without_ddp))
	logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

	eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

	if args.lr is None:  # only base_lr is specified
		args.lr = args.blr * eff_batch_size / 256

	logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
	logger.info("actual lr: %.2e" % args.lr)

	logger.info("accumulate grad iterations: %d" % args.accum_iter)
	logger.info("effective batch size: %d" % eff_batch_size)

	# build optimizer with layer-wise lr decay (lrd)
	param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
	                                    no_weight_decay_list=model_without_ddp.no_weight_decay(),
	                                    layer_decay=args.layer_decay
	                                    )

	optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	loss_scaler = NativeScaler()

	criterion = rc_loss_smooth

	logger.info("criterion = %s" % str(criterion))

	misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

	logger.info(f"Start training for {args.epochs} epochs")
	start_time = time.time()
	max_accuracy = 0.0
	model = DataParallel(model)

	for epoch in range(args.start_epoch, args.epochs):
		logger.info("-" * 20 + "epoch {}".format(epoch) + '-' * 20)

		_ = train_one_epoch(
			model, criterion, data_loader_train,
			optimizer, device, epoch, loss_scaler,
			args.clip_grad, mixup_fn,
			log_writer=logger,
			args=args,
			confidence=confidence
		)
		if epoch > 80:
			save_ckpt(args, model, epoch)

		test_stats = evaluate(data_loader_val, model, device, logger)
		logger.info(f"Accuracy of the network: {test_stats['acc1']:.1f}%")
		max_accuracy = max(max_accuracy, test_stats["acc1"])
		logger.info(f'Max accuracy: {max_accuracy:.2f}%')

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	main(args)
