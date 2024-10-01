#!/bin/bash

# Usage: ./generate_sbatch.sh modell visit cv_rate

# Arguments
MODELL=$1
VISIT=$2
cv_rate=$3

# Generate job name from modell, visit, and image_size
JOB_NAME="cv2_${cv_rate}_${VISIT}_${MODELL}"

# Create the SBATCH script with the given parameters
cat <<EOT > runrun_${JOB_NAME}.sh
#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH -N 2
#SBATCH -n 6
#SBATCH --gres=gpu:2
#SBATCH -t 4:00:00
#SBATCH -p dgx_normal_q
#SBATCH -A animalcv    
#SBATCH --mem=60G 
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
    --cv cv2 \\
    --cv_rate ${cv_rate} \\
    --batch_size 100 \\
    --epochs 300 \\
    --learning_rate 0.001 \\
    --trainable True \\
    --image_count_thr 10 \\
    --num_gpus 2 \\
    --opt Adam 

exit;
EOT

echo "SBATCH script runrun_${JOB_NAME}.sh generated."
