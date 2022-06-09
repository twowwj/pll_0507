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
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_vit

from engine_finetune_rc import train_one_epoch, evaluate
from torch.nn import DataParallel
from util.mlogger import create_logger
from datasets.affetnet_pll import get_dataset


def get_args_parser():
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
	# append
	parser.add_argument('--dataset', default='affectnet8', type=str)
	parser.add_argument('--data_path', default='/home/macww/hdd/elk3/small/', type=str, help='dataset path')
	parser.add_argument('--output_dir', default='/home/macww/hdd/elk3/small/results/', type=str)
	parser.add_argument('--exp', default='default', type=str)
	parser.add_argument('--nb_classes', default=7, type=int,
	                    help='number of the classification types')
	parser.add_argument('--smoothing', type=float, default=0.1,
	                    help='Label smoothing (default: 0.1)')
	parser.add_argument('--labelset', default='/data0/wwang/pll_0507/labelset_affectnet7/36558.pkl', type=str)
	#
	parser.add_argument('--batch_size', default=4, type=int,
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

	parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
	                    help='epochs to warmup LR')

	# Augmentation parameters
	parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
	                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
	parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
	                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')

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

def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss
	
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


def main(args):
	save_path = os.path.join(args.output_dir, args.exp)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	logger = create_logger(save_path)
	logger.info("{}".format(args).replace(', ', ',\n'))

	if args.dataset == 'affectnet7':
		args.nb_classes = 7
	elif args.dataset == 'affectnet8':
		args.nb_classes = 8

	data_loader_train, data_loader_val, train_givenY = get_dataset(args)
	tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
	confidence = train_givenY.float() / tempY
	confidence = confidence.cuda()

	device = torch.device(args.device)

	# fix the seed for reproducibility
	seed = args.seed + misc.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)

	cudnn.benchmark = True

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
	# criterion = rc_loss
	misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

	logger.info(f"Start training for {args.epochs} epochs")
	start_time = time.time()
	max_accuracy = 0.0

	model = DataParallel(model)

	for args.epoch in range(args.start_epoch, args.epochs):
		logger.info("-" * 20 + "epoch {}".format(args.epoch) + '-' * 20)

		one_epoch_best, confidence = train_one_epoch(
			model, criterion, data_loader_train,
			optimizer, device, args.epoch, loss_scaler,
			args.clip_grad, mixup_fn,
			log_writer=logger,
			args=args, data_loader_val=data_loader_val,
			confidence=confidence
		)

		logger.info(f"Accuracy of the network: {one_epoch_best:.1f}%")
		max_accuracy = max(max_accuracy, one_epoch_best)

		logger.info(f'Max accuracy: {max_accuracy:.2f}%')

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	main(args)
