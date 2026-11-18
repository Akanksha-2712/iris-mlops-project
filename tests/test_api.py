from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    r = client.get("/")
    assert r.status_code == 200

def test_prediction():
    r = client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2
    })
    assert r.status_code == 200
    assert r.json()["predicted_species"] in ["setosa", "versicolor", "virginica"]

def test_invalid_input():
    r = client.post("/predict", json={
        "sepal_length": -1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2
    })
    assert r.status_code == 422
