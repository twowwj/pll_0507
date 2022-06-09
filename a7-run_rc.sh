#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH -t 23:59:00
#SBATCH -c 10
#SBATCH -o /data/wjwang/pll_0507/maskfeat_rc_loss.txt
#SBATCH --gres gpu:1
#SBATCH --mem=96000

eval "$(conda shell.bash hook)"

bash

conda activate py37

cd /data/wjwang/pll_0507

python3 mae_finetune_rcloss_affectnet.py --exp 0510_mae599_rc --epochs 25 --dataset affectnet7 --data_path /data/wjwang/fer_dataset/small --batch_size 320 --warmup_epochs 5 --num_workers 16 --lr 1e-4 --finetune /data/wjwang/pll_0507/pretrain-MF/A7/checkpoint-599-hog.pth --output_dir /data/wjwang/pll_0507/log --labelset /data/wjwang/pll_0507/36558.pkl  &

wait
