1. run NER model
sbatch -A, --account=bsc48 --qos=acc_debug --cpus-per-task=20 --partition=acc --gres=gpu:1 run.slurm