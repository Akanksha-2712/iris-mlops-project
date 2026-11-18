
from pathlib import Path
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / "data" / "iris.csv"

    if not data.exists():
        raise FileNotFoundError("Run python src/prepare_data.py first")

    p = yaml.safe_load((ROOT / "params.yaml").read_text())
    df = pd.read_csv(data)

    cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    X, y = df[cols], df["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=float(p["test_size"]),
        random_state=int(p["random_state"]),
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        random_state=int(p["random_state"])
    )

    mlflow.set_experiment("iris-flower-classification")

    with mlflow.start_run():
        model.fit(Xtr, ytr)

        pred = model.predict(Xte)
        accuracy = float(accuracy_score(yte, pred))
        report = classification_report(yte, pred, output_dict=True)

        mlflow.log_params(p)
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict(report, "classification_report.json")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        out = ROOT / "artifacts"
        out.mkdir(exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "class_names": ["setosa", "versicolor", "virginica"]
            },
            out / "iris_model.joblib"
        )

        (out / "metrics.json").write_text(
            json.dumps(
                {"accuracy": accuracy, "report": report},
                indent=2
            )
        )

        print(f"Test accuracy: {accuracy:.4f}")
        print("Model and metrics saved successfully.")


if __name__ == "__main__":
    main()
