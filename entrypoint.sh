#!/bin/bash
set -e

# Activate conda environment
# source /anaconda/etc/profile.d/conda.sh
# conda activate fsfp

source /venv/bin/activate

# Start the FastAPI app
exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --log-config app/log_conf.yaml
