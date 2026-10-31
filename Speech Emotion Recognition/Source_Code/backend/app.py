import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from predict import predict_proba
from ser.config import LABELS


app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Predicts emotions from speech using our trained MFCC deep learning model.",
    version="1.0.0"
)


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


@app.get("/")
def home():
    return {
        "message": "Speech Emotion Recognition API is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict_emotion(file: UploadFile = File(...)):

    # 1. Validate the uploaded file
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload an audio file."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Upload WAV, MP3, FLAC, or OGG."
        )

    temp_path = None

    try:
        # 2. Save the uploaded audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # 3. Check whether the file is empty
        if os.path.getsize(temp_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty."
            )

        # 4. Use our existing trained model and preprocessing pipeline
        probabilities = predict_proba(temp_path)

        # 5. Find the emotion with the highest model score
        predicted_index = int(np.argmax(probabilities))
        predicted_emotion = LABELS[predicted_index]

        # 6. Prepare scores for all eight emotions
        emotion_scores = {
            label: round(float(score), 4)
            for label, score in zip(LABELS, probabilities)
        }

        # 7. Return the prediction
        return {
            "status": "success",
            "filename": file.filename,
            "predicted_emotion": predicted_emotion,
            "probabilities": emotion_scores
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Prediction error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the audio."
        )

    finally:
        # 8. Delete the temporary audio file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        file.file.close()