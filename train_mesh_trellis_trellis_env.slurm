#!/bin/bash
#SBATCH --job-name=train_synthetic_data
#SBATCH --partition=gpu_ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --account=deepl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jichen.guo@dfki.de
#SBATCH --chdir=/mnt/beegfs/home/jguo/projects/TRELLIS
#SBATCH --output=/mnt/beegfs/home/jguo/projects/TRELLIS/HPC_outputs/slurm_%j.out
#SBATCH --error=/mnt/beegfs/home/jguo/projects/TRELLIS/HPC_outputs/slurm_%j.err

set -euo pipefail

export KERAS_BACKEND=jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "============================================================"
echo "Starting training synthetic_data"
echo "Date      : $(date)"
echo "Node      : $(hostname)"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Partition : ${SLURM_JOB_PARTITION}"
echo "GPU       : ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================================"

# Optional: load modules if your HPC requires them
# module load cuda
# module load python

#PYTHON="${PYTHON:-python3}"

PROJECT_DIR="/mnt/beegfs/home/jguo/projects/paz"
# EXP_BASE="${PROJECT_DIR}/experiments_HPC/experiment_10"

#mkdir -p "${EXP_BASE}/checkpoints" "${EXP_BASE}/plots"

export XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_strict_conv_algorithm_picker=false \
--xla_gpu_autotune_level=0 \
--xla_gpu_enable_triton_gemm=false"

cd "${PROJECT_DIR}"

SIF="/mnt/beegfs/home/jguo/images/trellis.sif"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not_set}"

nvidia-smi
DATASET=/mnt/beegfs/home/jguo/datasets/synthetic_datasets/synthetic_rgbd_1000_v4

singularity exec --nv --cleanenv \
  --bind /mnt/beegfs:/mnt/beegfs \
  --env PYTHONNOUSERSITE=1 \
  --env ATTN_BACKEND=flash-attn \
  --env SPCONV_ALGO=native \
  "${SIF}" \
  /opt/conda/bin/python /mnt/beegfs/home/jguo/projects/TRELLIS/run_trellis_exchange.py \
  --exchange "${PROJECT_DIR}/experiments/trellis_exchange_test"\
  --output "${PROJECT_DIR}/experiments/trellis_exchange_test_results"\
  --max-samples 2

EXIT_CODE=$?


exit ${EXIT_CODE}