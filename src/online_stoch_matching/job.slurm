#!/bin/bash
#SBATCH --partition=smp
#SBATCH --cluster=smp
#SBATCH --job-name=Experiment_JobArray     # create a short name for your job
#SBATCH --nodes=1                               # node count
#SBATCH --ntasks-per-node=1                     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G                        # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=3-00:00:00                         # total run time limit (HH:MM:SS)
#SBATCH --array=0-19                             # job array with index values 0, 1, 2, 3, 4, 5, 6, 7, 8, 9



echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)


module purge
 
module load gcc/8.2.0
module load python/anaconda3.9-2021.11
 
source activate py39
 
srun python experiments.py
