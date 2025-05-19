#!/bin/bash
#SBATCH --job-name=ner_on_llm_predictor    # Job name
#SBATCH --output=ner_on_llm_%j.out         # Output file (%j = job ID)
#SBATCH --error=ner_on_llm_%j.err          # Error file
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks



# Set Python path
PYTHON_EXEC=/apps/ACC/PYTHON/3.12.1/INTEL/bin/python

# Check Python version
echo "Using Python at: $PYTHON_EXEC"
$PYTHON_EXEC --version


# Load CUDA module (match nvidia-smi CUDA version)
module purge
module load oneapi hdf5 python cuda/12.2


# Verify GPU availability
$PYTHON_EXEC -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set working directory
cd /gpfs/home/bsc/bsc148007/projects/NERonLLM

# Run the Python script
$PYTHON_EXEC v2.py

# Print resource usage
echo "Job completed at $(date)"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,MaxRSS,Elapsed