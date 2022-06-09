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
      "python3 mae_finetune_pico_affectnet.py --epochs 25 "
      "--exp 0507_mae599_pico "
      "--dataset affectnet7 "
      "--data_path /data/wjwang/fer_dataset/small "
      "--output_dir /data/wjwang/pll_0507/log "
      "--batch_size 640 "
      "--finetune /data/wjwang/checkpoint-599.pth "
	  "--labelset /data/wjwang/pll_0507/36558.pkl"
   )

   # run_msg(
   #    "python3 mae_finetune_pico_affectnet.py --epochs 25 "
   #    "--exp 0507_maskfeat599_pico "
   #    "--dataset affectnet7 "
   #    "--data_path /data/wjwang/fer_dataset/small "
   #    "--output_dir /data/wjwang/pll_0507/log "
   #    "--batch_size 640 "
   #    "--finetune /data/wjwang/checkpoint-599-hog.pth "
	#   "--labelset /data/wjwang/pll_0507/36558.pkl"
   # )

   # run_msg(
   #    "python3 mae_finetune_pico_affectnet.py --epochs 25 "
   #    "--exp 0507_mae599_pico "
   #    "--dataset affectnet7 "
   #    "--data_path /raid/libraboli/FER/small/ "
   #    "--output_dir /raid/libraboli/FER/log_results "
   #    "--batch_size 640 "
   #    "--finetune /raid/libraboli/FER/pretrain_models/MAE/checkpoint-599.pth"
   # )

except:
   pass