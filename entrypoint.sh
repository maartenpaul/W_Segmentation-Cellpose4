#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
if [ "$TRAINING_MODE" = "true" ]; then
    echo "Training mode enabled, launching train.py..."
    conda activate cellpose_env
    exec python /app/train.py "$@"
else
    conda activate cytomine_py37
    exec python /app/run.py "$@"
fi
