import hashlib
import numpy as np
import pandas as pd

from pathlib import Path
from lightgbm import LGBMRegressor

DATASET = "data/processed/kcat.csv"
FEAT_DIR = Path("data/msa2d/features")


def pid(seq):
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


df = pd.read_csv(DATASET)

X=[]
y=[]

for _,row in df.iterrows():

    p = FEAT_DIR / f"{pid(row.sequence)}.npy"

    if not p.exists():
        continue

    X.append(np.load(p))
    y.append(row.target)

X=np.array(X)
y=np.array(y)

model=LGBMRegressor(
    n_estimators=500,
    random_state=42
)

model.fit(X,y)

for i,v in sorted(
    enumerate(model.feature_importances_),
    key=lambda x:x[1],
    reverse=True
):
    print(i,v)

