from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))
    out = ROOT / "data" / "iris.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} observations to {out}")

if __name__ == "__main__":
    main()
