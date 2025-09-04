#!/bin/bash
#SBATCH --job-name=tf_no_em_gs
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-elbo
#SBATCH --ntasks=1
#SBATCH --time=2000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5         # Number of CPU cores per task
#SBATCH --output=slurm_default/out/%x_device_used-%N-%u-%j.out
#SBATCH --error=slurm_default/err/%x_device_used-%N-%u-%j.err 
#SBATCH --mail-type=ALL,TIME_LIMIT_80

cd /home/elhanan/home/shirabaum/MLHC_project # SBATCH --array=0-5%6

# JOB_NAME="$1"

# echo "Running job: $JOB_NAME"

# # Create log directories
mkdir -p "runs/slurm/out/"
mkdir -p "runs/slurm/err/"

# MODELS=(
# d
# y
# x
# m0
# m2
# m
# )
# MODEL_ID=${MODELS[$SLURM_ARRAY_TASK_ID]}
SRUN_OUT="runs/slurm/out/device_used-${SLURMD_NODENAME}-${SLURM_SUBMIT_HOST}-${SLURM_JOB_ID}.out"
SRUN_ERR="runs/slurm/err/device_used-${SLURMD_NODENAME}-${SLURM_SUBMIT_HOST}-${SLURM_JOB_ID}.err"

# # Run the specified Python script
# srun /home/elhanan/PROGRAMS/conda/envs/evo_model/bin/python ../scripts/"$JOB_NAME" > "$SRUN_OUT" 2> "$SRUN_ERR"
# lps_females_anonymous   sample_data_formatted
srun /home/elhanan/PROGRAMS/conda/envs/shiri_gpu/bin/python3 main.py --run-name tf_no_te_arch_gs --num-epochs 27 --model-type transformer --dataset-name lps_females_anonymous --device-name gpu --grid-search --grid-search-params param_grids/params_trans_no_embed_arch.json > "$SRUN_OUT" 2> "$SRUN_ERR"
