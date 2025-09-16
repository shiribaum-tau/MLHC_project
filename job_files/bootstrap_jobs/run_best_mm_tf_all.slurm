#!/bin/bash
#SBATCH --job-name=best_aupr_mm_tf_bc_all
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
# lps_females_anonymous   sample_data_formatted lps_individuals_anonymous
srun /home/elhanan/PROGRAMS/conda/envs/shiri_gpu/bin/python3 main.py --run-name best_aupr_mm_tf_bc_all --num-epochs 100 --dropout 0 --weight-decay 0.0001 --time-embed-dim 8 --hidden-dim 32 --num-layers 4 --num-heads 8 --model-type mm_transformer --dataset-name lps_individuals_anonymous_w_blood --device-name gpu --bootstrap-test --model-to-load-dir runs/best_aupr_mm_tf_bc_all/log --base-output-dir bootstrap_test_runs > "$SRUN_OUT" 2> "$SRUN_ERR"
