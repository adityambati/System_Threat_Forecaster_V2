import os
import sys
import io
import pandas as pd
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from pipeline.predict_pipeline import PredictPipeline
from engine.logger import logger

app = FastAPI(title="System Threat Forecaster V2")

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

# Initialize Predict Pipeline
predict_pipeline = PredictPipeline()


@app.get("/")
async def home(request: Request):
    """Renders the dashboard upload page."""
    return templates.TemplateResponse(request, "index.html", {"error_message": None})


@app.post("/predict")
async def predict_file(request: Request, file: UploadFile = File(...)):
    """Handles CSV upload, calculates threat probabilities, and returns results dashboard."""
    try:
        logger.info(f"Received file for threat analysis: {file.filename}")

        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))

        # Preserve MachineID or create index identifier
        id_col = predict_pipeline.config["data"]["id_column"]
        if id_col in input_df.columns:
            machine_ids = input_df[id_col].tolist()
        else:
            machine_ids = [f"Machine_{i+1}" for i in range(len(input_df))]

        # Predict threat probabilities
        probabilities = predict_pipeline.predict(input_df)

        results_df = pd.DataFrame(
            {
                "MachineID": machine_ids,
                "Threat_Probability": probabilities,
                "Threat_Status": [
                    "High Risk" if p >= 0.35 else "Clean" for p in probabilities
                ],
            }
        )

        total_systems = len(results_df)
        high_risk_count = int((results_df["Threat_Probability"] >= 0.35).sum())
        clean_count = total_systems - high_risk_count
        risk_percentage = (
            round((high_risk_count / total_systems) * 100, 2)
            if total_systems > 0
            else 0
        )

        # Save temporary results CSV for download
        os.makedirs("artifacts/predictions", exist_ok=True)
        temp_csv_path = "artifacts/predictions/latest_predictions.csv"
        results_df.to_csv(temp_csv_path, index=False)

        preview_data = results_df.head(20).to_dict(orient="records")
        all_records = results_df.to_dict(orient="records")

        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "filename": file.filename,
                "total_systems": total_systems,
                "high_risk_count": high_risk_count,
                "clean_count": clean_count,
                "risk_percentage": risk_percentage,
                "preview_data": preview_data,
                "all_records": all_records,
            },
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return templates.TemplateResponse(
            request, "index.html", {"error_message": f"Error processing file: {str(e)}"}
        )


@app.get("/download-predictions")
async def download_predictions():
    file_path = "artifacts/predictions/latest_predictions.csv"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, media_type="text/csv", filename="threat_predictions.csv"
        )
    return {"error": "Prediction file not found. Run a scan first."}
