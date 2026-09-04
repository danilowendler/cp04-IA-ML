import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import ARTIFACTS_DIR, MLFLOW_TRACKING_URI

URI_FILE = ARTIFACTS_DIR / "best_model_uri.txt"

_model = None

app = FastAPI(
    title="IMDB Sentiment API",
    version="1.0.0",
    description="API local de predição usando MLP + TF-IDF + MLflow.",
)


class PredictionRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="Avaliação de filme em texto.",
    )

    @field_validator("review")
    @classmethod
    def review_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("A review deve conter texto.")
        return value


def load_model():
    global _model

    if _model is not None:
        return _model

    if not URI_FILE.exists():
        raise RuntimeError(
            "Modelo não encontrado. Execute primeiro: python src/train.py"
        )

    model_uri = URI_FILE.read_text(encoding="utf-8").strip()

    # O treinamento e a API devem utilizar o mesmo backend do MLflow.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    _model = mlflow.sklearn.load_model(model_uri)

    return _model


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="O modelo vencedor ainda não está disponível.",
        ) from exc


@app.post("/predict")
def predict(payload: PredictionRequest):
    try:
        model = load_model()

        prediction = model.predict([payload.review])[0]

        probabilities = model.predict_proba([payload.review])[0]
        classes = list(model.classes_)

        probability = float(
            probabilities[classes.index(prediction)]
        )

        return {
            "prediction": str(prediction),
            "probability": round(probability, 4),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Não foi possível realizar a predição.",
        ) from exc
