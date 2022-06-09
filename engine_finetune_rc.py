# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, data_loader_val=None, confidence=None):
	model.train(True)
	metric_logger = misc.MetricLogger(delimiter="  ")
	metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
	header = 'Epoch: [{}]'.format(epoch)
	print_freq = 20

	accum_iter = args.accum_iter

	optimizer.zero_grad()

	one_epoch_best = 0
	for data_iter_step, (samples, labelset, targets, index) in enumerate(
			metric_logger.log_every(data_loader, print_freq, header, log_writer)):

		# we use a per iteration (instead of per epoch) lr scheduler
		if data_iter_step % accum_iter == 0:
			lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

		samples = samples.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		labelset = labelset.to(device, non_blocking=True)

		if mixup_fn is not None:
			samples, targets = mixup_fn(samples, targets)

		with torch.cuda.amp.autocast():
			outputs = model(samples)
			loss = criterion(outputs, confidence, index, num_classes=args.nb_classes, lb_smooth=args.smoothing)

		loss_value = loss.item()

		if not math.isfinite(loss_value):
			log_writer.info("Loss is {}, stopping training".format(loss_value))
			sys.exit(1)

		loss /= accum_iter
		loss_scaler(loss, optimizer, clip_grad=max_norm,
		            parameters=model.parameters(), create_graph=False,
		            update_grad=(data_iter_step + 1) % accum_iter == 0)
		if (data_iter_step + 1) % accum_iter == 0:
			optimizer.zero_grad()

		torch.cuda.synchronize()

		metric_logger.update(loss=loss_value)
		min_lr = 10.
		max_lr = 0.
		for group in optimizer.param_groups:
			min_lr = min(min_lr, group["lr"])
			max_lr = max(max_lr, group["lr"])

		metric_logger.update(lr=max_lr)

		if data_loader_val is not None:
			if data_iter_step % 100 == 0 and data_iter_step > 0:
				test_stats = evaluate(data_loader_val, model, device, log_writer)
				one_epoch_best = max(one_epoch_best, test_stats['acc1'])

		confidence = confidence_update(model, confidence, samples, labelset, index)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	log_writer.info("Averaged stats : {}".format(metric_logger))
	return one_epoch_best, confidence


@torch.no_grad()
def evaluate(data_loader, model, device, logger):
	criterion = torch.nn.CrossEntropyLoss()

	metric_logger = misc.MetricLogger(delimiter="  ")
	header = 'Test:'

	# switch to evaluation mode
	model.eval()

	for batch in metric_logger.log_every(data_loader, 10, header, logger=logger):
		images = batch[0]
		target = batch[-1]
		images = images.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

		# compute output
		with torch.cuda.amp.autocast():
			output = model(images)
			loss = criterion(output, target)

		acc1, acc2, acc5 = accuracy(output, target, topk=(1, 2, 5))

		batch_size = images.shape[0]
		metric_logger.update(loss=loss.item())
		metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
		metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
		metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
	            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def confidence_update(model, confidence, batchX, batchY, batch_index):
	with torch.cuda.amp.autocast():
		batch_outputs = model(batchX)
		temp_un_conf = F.softmax(batch_outputs, dim=1)
		confidence[batch_index, :] = temp_un_conf * batchY
		base_value = confidence.sum(dim=1).unsqueeze(1).repeat(
			1, confidence.shape[1])
		confidence = confidence / base_value
	return confidence
