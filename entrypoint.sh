#!/bin/bash
set -e

# Activate conda environment
source /anaconda/etc/profile.d/conda.sh
conda activate fsfp

# Start the FastAPI app
exec uvicorn app.main:app --host 0.0.0.0 --port 80 --log-config log_conf.yaml
