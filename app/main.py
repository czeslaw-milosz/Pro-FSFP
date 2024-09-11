import logging
import subprocess
import os

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import app.constants as app_constants
from app.models import MutantRequestData


with open("app/api_description.md", "r") as f:
    description_md = "".join(line for line in f.readlines())
app = FastAPI(
    title="Zika B-cell score API",
    description=description_md
)


@app.get("/")
async def root():
    return {"message": "Do not go gentle into that good night"}

@app.post("/predict")
async def predict(data: MutantRequestData):
    target_protein_id = data.target_protein_id
    checkpoint = app_constants.CHECKPOINT_MAPPING[data.checkpoint]
    task_name = data.task_name
    device = data.device
    logging.info(f"Current working dir: {os.getcwd()}")

    # Convert input data to DataFrame
    df = pd.DataFrame({
        "mutant": data.mutant,
        "mutated_sequence": data.mutated_sequence
    })
    
    # Save DataFrame to CSV in FSFP's inputs data directory
    os.makedirs(app_constants.INPUT_DIR, exist_ok=True)
    file_path = f"{app_constants.INPUT_DIR}/{target_protein_id}_{task_name}_input_data.csv"
    df.to_csv(file_path, index=False)
    logging.info(f"Stored input data with {len(df)} rows at {file_path}")

    # Convert the input data to ProteinGym format
    subprocess.run(["python", "envelope2protein.py", "--input_file", file_path])
    logging.info(f"Converted input data to ProteinGym format and saved at {file_path}")

    # Run FSFP's prerocessing routine
    subprocess.run(["python", "preprocess.py"])
    assert os.path.exists("data/merged.pkl"), "Preprocessed file not found"
    logging.info("Preprocessing completed successfully")

    # Predict
    logging.info("Starting prediction")
    pred_command = ["python", "main.py", "-ckpt", checkpoint, 
                    "--model", "esm2", "--protein", target_protein_id, "--predict"]
    if device == "cpu":
        pred_command.append("--force_cpu")
    logging.info(f"Using device: {device}")
    subprocess.run(pred_command)
    assert os.path.exists(app_constants.OUTPUT_FNAME), "Prediction file not found, something went wrong"

    # Return result
    result = pd.read_csv(app_constants.OUTPUT_FNAME)
    return JSONResponse(content={
        "message": "Prediction pipeline completed succesfullly.",
        "mutant": result["mutant"].tolist(),
        "prediction": result["prediction"].tolist(),
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
