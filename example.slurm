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


echo "============================================================"
echo "TRELLIS"
echo "Date      : $(date)"
echo "Node      : $(hostname)"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Partition : ${SLURM_JOB_PARTITION}"
echo "GPU       : ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================================"


singularity exec --nv --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --env ATTN_BACKEND=flash-attn \
    --env SPCONV_ALGO=native \
    ~/images/trellis.sif \
    /opt/conda/bin/python example_time.py

EXIT_CODE=$?


exit ${EXIT_CODE}