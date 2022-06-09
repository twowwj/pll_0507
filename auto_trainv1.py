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
	run_msg(
		"python3 mae_finetune_base_affectnet.py "
		"--exp 0507_maskfeat599_base "
		"--dataset affectnet7 "
		"--data_path /raid/libraboli/FER/small/ "
		"--output_dir /raid/libraboli/FER/log_results "
		"--batch_size 640 "
		"--finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40//checkpoint-599.pth"
	)

	run_msg(
		"python3 mae_finetune_rcloss_affectnet.py "
		"--exp 0507_maskfeat599_rc "
		"--dataset affectnet7 "
		"--data_path /raid/libraboli/FER/small/ "
		"--output_dir /raid/libraboli/FER/log_results "
		"--batch_size 640 "
		"--finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40//checkpoint-599.pth --labelset "
	)


	run_msg(
		"python3 mae_finetune_base.py "
		"--exp 0507_mae599_base "
		"--dataset affectnet7 "
		"--data_path /raid/libraboli/FER/small/ "
		"--output_dir /raid/libraboli/FER/log_results "
		"--batch_size 640 "
		"--finetune /raid/libraboli/FER/pretrain_models/MAE//checkpoint-599.pth"
	)

	run_msg(
		"python3 mae_finetune_rcloss_affectnet.py "
		"--exp 0507_mae599_rc "
		"--dataset affectnet7 "
		"--data_path /raid/libraboli/FER/small/ "
		"--output_dir /raid/libraboli/FER/log_results "
		"--batch_size 640 "
		"--finetune /raid/libraboli/FER/pretrain_models/MAE/checkpoint-599.pth"
	)

except:
	pass
