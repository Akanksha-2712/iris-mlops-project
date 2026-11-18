# Iris Flower Classification using MLOps

Classifies Iris flowers from four measurements (cm) into setosa, versicolor, or virginica.

## Tools
Git, DVC, scikit-learn, MLflow, pytest, GitHub Actions, FastAPI, Docker.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare data and train
```bash
python src/prepare_data.py
python src/train.py
```
The built-in scikit-learn Iris dataset has 150 observations (50 per species).

## MLflow
```bash
mlflow ui
```
Open http://127.0.0.1:5000 to view the logged experiment.

## DVC
```bash
git init
dvc init
dvc repro
```
DVC pipeline stages are defined in `dvc.yaml`.

## Tests
```bash
pytest -v
```

## API
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs. POST to `/predict`:
```json
{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
```

## Docker
```bash
docker build -t iris-mlops .
docker run -p 8000:8000 iris-mlops
```
Then visit http://127.0.0.1:8000/docs.

GitHub Actions runs data preparation, model training, and pytest on pushes and pull requests.
