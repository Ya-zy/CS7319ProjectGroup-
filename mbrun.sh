#!/bin/bash
#SBATCH -J mnist_lstm_test
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
module load anaconda2/5.3.0 cuda/10.0 cudnn/7.4.2 cmake/3.13.0 gcc/7.3.0 make/4.2.1
source activate pytorch1.0.0
python -u  main.py --config res34_256x192_d256x3_adam_lr1e-3.yaml