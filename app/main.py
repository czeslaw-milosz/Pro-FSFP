import os

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Pro-FSFP API"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # Save the uploaded file
    file_path = f"uploads/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Read the CSV file as a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # You can perform operations on the DataFrame here
    
    return JSONResponse(content={
        "message": "File uploaded and processed successfully",
        "filename": file.filename,
        "row_count": len(df)
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)