#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --job-name=ud-gnn
#SBATCH --time=1-0

module purge
module load cuda
module load anaconda3

set -e

source activate meshcnn

## run the training
python train.py \
--dataroot datasets/human_seg \
--name human_seg \
--arch mconvnet \
--dataset_mode classification \
--ncf 32 64 128 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 32 \
--lr 0.0002 \
--num_aug 20 \
--slide_verts 0.2 \
--fc_n 128 \
--niter 100 \
--niter_decay 0 \
--norm group \
--num_groups 16 \
--flip_edges 0.05 \
--no_vis
