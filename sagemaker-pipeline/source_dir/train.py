import pandas as pd
import joblib
from xgboost import XGBClassifier

if __name__ == "__main__":
    # read the two-column CSV: first col is label, next 4 are features
    df = pd.read_csv("train.csv", header=None)
    X = df.drop(0, axis=1)
    y = df[0].astype(int)

    model = XGBClassifier(n_estimators=5, eval_metric="logloss")
    model.fit(X, y)
    joblib.dump(model, "model.joblib")
