import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

DATASET = "data/master/kcat.csv"

df = pd.read_csv(DATASET)

def pid(seq):
    return hashlib.sha256(
        str(seq).encode()
    ).hexdigest()[:16]

feat_dir = Path("data/msa2d/features")

X = []
y = []

for _, row in df.iterrows():

    protein_id = pid(row["sequence"])

    fp = feat_dir / f"{protein_id}.npy"

    if not fp.exists():
        continue

    X.append(np.load(fp))
    y.append(row["target"])

X = np.array(X)
y = np.array(y)

n = int(len(X) * 0.8)

X_train = X[:n]
X_test = X[n:]

y_train = y[:n]
y_test = y[n:]

model = LGBMRegressor(
    n_estimators=500,
    random_state=42,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(
    "R2 =",
    r2_score(y_test, pred)
)
