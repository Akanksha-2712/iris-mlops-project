from pathlib import Path
import joblib, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "iris_model.joblib"
app = FastAPI(title="Iris Flower Classifier")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/")
def home():
    return {"message": "Iris Flower Classification API", "docs": "/docs"}

@app.post("/predict")
def predict(data: IrisInput):
    if not MODEL_PATH.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Train the model first: python src/train.py")
    bundle = joblib.load(MODEL_PATH)
    x = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    model = bundle["model"]
    idx = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    names = bundle["class_names"]
    return {"predicted_species": names[idx],
            "probabilities": {n: round(float(v), 4) for n, v in zip(names, probs)}}
