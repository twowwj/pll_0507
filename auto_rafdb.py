import os
import shutil
import torch
from torchvision.models.resnet import resnet152
from torch.nn import DataParallel
import time
import sys


def run_msg(msg):
	print(msg)
	os.system(msg)
	time.sleep(3)


try:

	for epoch in [1, 20, 25, 30, 35, 40, 45, 50]:
		print(
			"python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py "
			"--exp 0511_rafdb_rcloss_{} --epochs {} "
			"--dataset rafdb "
			"--data_path /raid/libraboli/FER/raf_0806/ "
			"--output_dir /raid/libraboli/FER/log_results "
			"--batch_size 160 --num_workers 12 --warmup_epochs 0 "
			"--finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth "
			"--labelset /raid/libraboli/FER/small/0113_0054.pkl".format(epoch, epoch)
		)

		print(
			"python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py "
			"--exp 0511_rafdb_pico_{} --epochs {} "
			"--dataset rafdb "
			"--data_path /raid/libraboli/FER/raf_0806/ "
			"--output_dir /raid/libraboli/FER/log_results "
			"--batch_size 160 --num_workers 12 --warmup_epochs 0 "
			"--finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth "
			"--labelset /raid/libraboli/FER/small/0113_0054.pkl".format(epoch, epoch)
		)



except:
	pass

# model = resnet152().cuda()
# model = DataParallel(model)
# while True:
# 	model(torch.randn(64, 3, 512, 512))
