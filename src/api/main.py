"""
API REST d'inférence pour la détection de rétinopathie diabétique.

Endpoints :
    GET  /health            — Health check
    POST /predict           — Prédiction sur une image unique
    POST /predict/batch     — Prédiction batch (jusqu'à 10 images)
    GET  /model/info        — Informations sur le modèle chargé
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.data.preprocessing import PreprocessingConfig, RetinalImagePreprocessor

logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/optimized/model_dynamic.tflite")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "512"))

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
CLINICAL_ACTIONS = {
    0: "Aucune action requise — contrôle annuel",
    1: "Surveillance rapprochée — contrôle dans 6 mois",
    2: "Référer à l'ophtalmologue — évaluation dans 3 mois",
    3: "Référer en urgence — consultation ophtalmologique sous 1 mois",
    4: "Référer en urgence — consultation spécialisée immédiate",
}


# ---- Modèles Pydantic ----

class PredictionResult(BaseModel):
    grade: int
    grade_name: str
    confidence: float
    probabilities: dict
    clinical_action: str
    inference_time_ms: float


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    total_inference_time_ms: float


class ModelInfo(BaseModel):
    model_path: str
    input_size: int
    num_classes: int
    model_size_mb: float


# ---- Chargement du modèle ----

class ModelServer:
    """Serveur d'inférence TFLite optimisé."""

    def __init__(self, model_path: str, input_size: int):
        self.input_size = input_size
        self.preprocessor = RetinalImagePreprocessor(
            PreprocessingConfig(
                target_size=input_size,
                apply_clahe=True,
                apply_ben_graham=True,
            )
        )

        # Chargement TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        logger.info(
            "Modèle chargé — %s (%.1f MB), input: %s",
            model_path,
            self.model_size_mb,
            self.input_details[0]["shape"],
        )

    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Prétraite une image depuis des bytes bruts."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Impossible de décoder l'image")

        # Pipeline de preprocessing
        image = self.preprocessor.extract_circular_roi(image)
        image = self.preprocessor.apply_ben_graham_normalization(image)
        image = self.preprocessor.apply_clahe_enhancement(image)
        image = self.preprocessor.resize_and_normalize(image)

        return image

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Exécute l'inférence TFLite."""
        input_data = np.expand_dims(image, 0).astype(
            self.input_details[0]["dtype"]
        )
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output[0]


# ---- Application FastAPI ----

model_server: Optional[ModelServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle au démarrage."""
    global model_server
    logger.info("Chargement du modèle depuis %s...", MODEL_PATH)
    model_server = ModelServer(MODEL_PATH, INPUT_SIZE)
    yield
    logger.info("Arrêt du serveur")


app = FastAPI(
    title="RetinAI — API de détection de rétinopathie",
    description="API d'inférence pour la classification de rétinopathie diabétique "
                "à partir d'images de fond d'œil.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Vérification de l'état du service."""
    return {"status": "healthy", "model_loaded": model_server is not None}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Informations sur le modèle déployé."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return ModelInfo(
        model_path=MODEL_PATH,
        input_size=INPUT_SIZE,
        num_classes=len(CLASS_NAMES),
        model_size_mb=round(model_server.model_size_mb, 2),
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)):
    """
    Prédiction sur une image de fond d'œil.
    
    Accepte : PNG, JPEG, TIFF, BMP.
    Retourne : grade de rétinopathie, confiance et action clinique.
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Lecture et preprocessing
    contents = await image.read()
    start = time.perf_counter()

    try:
        processed = model_server.preprocess_from_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Inférence
    probabilities = model_server.predict(processed)
    inference_time = (time.perf_counter() - start) * 1000

    grade = int(np.argmax(probabilities))
    confidence = float(probabilities[grade])

    return PredictionResult(
        grade=grade,
        grade_name=CLASS_NAMES[grade],
        confidence=round(confidence, 4),
        probabilities={
            CLASS_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
        clinical_action=CLINICAL_ACTIONS[grade],
        inference_time_ms=round(inference_time, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(images: List[UploadFile] = File(...)):
    """Prédiction batch (max 10 images)."""
    if model_server is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images par requête")

    start_total = time.perf_counter()
    predictions = []

    for image in images:
        contents = await image.read()
        start = time.perf_counter()

        try:
            processed = model_server.preprocess_from_bytes(contents)
            probabilities = model_server.predict(processed)
        except ValueError:
            predictions.append(
                PredictionResult(
                    grade=-1, grade_name="Error", confidence=0.0,
                    probabilities={}, clinical_action="Erreur de traitement",
                    inference_time_ms=0.0,
                )
            )
            continue

        inference_time = (time.perf_counter() - start) * 1000
        grade = int(np.argmax(probabilities))

        predictions.append(
            PredictionResult(
                grade=grade,
                grade_name=CLASS_NAMES[grade],
                confidence=round(float(probabilities[grade]), 4),
                probabilities={
                    CLASS_NAMES[i]: round(float(p), 4)
                    for i, p in enumerate(probabilities)
                },
                clinical_action=CLINICAL_ACTIONS[grade],
                inference_time_ms=round(inference_time, 2),
            )
        )

    total_time = (time.perf_counter() - start_total) * 1000

    return BatchPredictionResult(
        predictions=predictions,
        total_inference_time_ms=round(total_time, 2),
    )
