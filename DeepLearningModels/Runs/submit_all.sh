#!/bin/bash

for file in runrun_*.sh; do
    if [[ $file = *"runrun"* ]]; then
        sbatch "$file"
    fi
done
