#!/bin/bash

# Arguments
MODELL=$1
VISIT=$2

# Generate job name from modell, visit, and image_size
JOB_NAME="${VISIT}_${MODELL}"

# Create the SBATCH script with the given parameters
cat <<EOT > runrun_${JOB_NAME}.sh
#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH -N 2
#SBATCH -n 6
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -p dgx_normal_q
#SBATCH -A animalcv    
#SBATCH --output=job_output_%x_%j.log
#SBATCH --error=job_error_%x_%j.log

module load apps site/tinkercliffs/easybuild/setup
module load Anaconda3/2024.02-1
source activate tf_gpu3
module list

python /home/yebi/ComputerVision_PLF/Pig_BW/Pig_BW_DL_beta/DL/RunRun4pred_kcam.py \\
    --modell ${MODELL} \\
    --image_size 150 \\
    --visit ${VISIT} \\
    --batch_size 100 \\
    --epochs 300 \\
    --learning_rate 0.001 \\
    --trainable True \\
    --image_count_thr 20 \\
    --num_gpus 2 \\
    --opt Adam 

exit;
EOT

echo "SBATCH script runrun_${JOB_NAME}.sh generated."
