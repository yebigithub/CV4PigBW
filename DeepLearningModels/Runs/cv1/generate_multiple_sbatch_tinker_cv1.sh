#!/bin/bash

# List of models
models=(
    "MobileViT_V2_200"
    "MobileViT_V2_150"
    "MobileViT_V2_100"
    "MobileNetV3Large100"
    "MobileViT_XXS"
    "MobileViT_V2_050"
    "MobileViT_S"
    "MobileNet050"
    "MobileNet075"
    "MobileNetV3Large075"
    "MobileNet100"
    "MobileNetV3Small050"
    "ResNet50"
)

# List of visits
visits=(
    '0718'
    "0801"
    "0815"
    "0829"
    "0912"
    "0927"
)

cv_rates=(
    0.8
)

# Loop through each model and visit combination
for model in "${models[@]}"
do
    for visit in "${visits[@]}"
    do
        # Optionally, loop through cross-validation rates (if needed)
        for cv_rate in "${cv_rates[@]}"
        do
            ./generate_sbatch_tinker_cv1.sh "$model" "$visit"
        done
    done
done

echo "All jobs generated."
