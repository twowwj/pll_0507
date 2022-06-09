python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 1 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_1

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 1 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_1

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 1 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_1

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 1 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_1

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 20 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_20

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 20 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_20

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 20 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_20

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 20 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_20

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 25 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_25

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 25 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_25

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 25 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_25

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 25 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_25

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 30 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_30

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 30 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_30

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 30 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_30

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 30 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_30

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 35 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_35

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 35 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_35

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 35 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_35

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 35 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_35

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 40 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_40

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 40 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_40

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 40 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_40

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 40 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_40

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 45 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_45

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 45 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_45

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 45 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_45

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 45 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_45

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 50 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_rcloss_50

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 50 --dataset rafdb --data_path /raid/libraboli/FER/raf_0806/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/rafdb/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0113_0054.pkl --exp 0511_rafdb_pico_50

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_rcloss.py --epochs 50 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_rcloss_50

python3 -m torch.distributed.launch --nproc_per_node=4 main_finetune_pico.py --epochs 50 --dataset ferplus --data_path /raid/libraboli/FER/alignment3/ --output_dir /raid/libraboli/FER/log_results --batch_size 160 --num_workers 12 --warmup_epochs 0 --finetune /raid/libraboli/FER/pretrain_models/MaskFeat_40/ferplus/checkpoint-599.pth --labelset /raid/libraboli/FER/small/0415_0415.pkl --exp 0511_ferplus_pico_50
