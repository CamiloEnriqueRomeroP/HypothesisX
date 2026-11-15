"""
PAPER SYN_12+X1 comparition with benchmark (SYN_12) and with the original 3-class spec (SYN_3CLASS2).
============================
"""

import sys
import time
from pathlib import Path

# Configuración robusta y dinámica de rutas para la carpeta src de HypothesisX
current_dir = Path(__file__).resolve().parent
src_candidates = [current_dir / "src", current_dir.parent / "src", current_dir / "HypothesisX" / "src"]

for candidate in src_candidates:
    if candidate.exists() and (candidate / "SearchLibrium").exists():
        sys.path.insert(0, str(candidate))
        break

import numpy as np
import pandas as pd
import random

random.seed(20)

from SearchLibrium.latent_class_final import LatentClassFinal

# =====================================================================
# DATA
# =====================================================================
df = pd.read_csv("SYNTHETIC_3CLASS2.csv")

AS_VARS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7_irr", "X8_irr"]
IS_VARS = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8_irr", "Z9_irr"]
ALL_VARS = AS_VARS + IS_VARS

X = df[ALL_VARS].values
y = df["CHOICE"].values.astype(float)
ids = df["Cho_ID"].values
ind_ids = df["Ind_ID"].values
alts = df["alt"]
av = df["AV"]


# =====================================================================
### BENCHMARK SPECIFICATION ###
# =====================================================================
class_params_spec = {
    "Class_1": ["X1", "X2", "X3", "X6"],
    "Class_2": ["X1", "X2", "X4", "X6"],
    "Class_3": ["X1", "X2", "X5"],
}

member_params_spec = {
    "Class_1": ["Z1", "Z4", "Z5"],
    "Class_2": ["Z2", "Z4", "Z6"],
    "Class_3": ["Z3", "Z7"],
}

BASE_CLASS_NEW = 3  # 1-indexed (para LatentClassFinal)


# =====================================================================
# WARM START FROM SYN_12
# =====================================================================
betas0 = [
    # Class 1 (12.58%)
    {"X2": 1.1921, "X3": -0.0262, "X4": -1.3334, "X6": -0.9892, "X7_irr": -0.1569, "X8_irr": -0.1815},
    # Class 2 (6.72%)
    {
        "X1": -0.5538,
        "X2": 1.5606,
        "X3": -0.1208,
        "X4": -1.5175,
        "X5": 0.0548,
        "X6": -1.1505,
        "X7_irr": -0.2433,
        "X8_irr": 0.0504,
    },
    # Class 3 (52.78%)
    {"X1": -0.9570, "X2": 0.7446, "X3": 1.4801, "X4": -0.0109, "X6": 0.8479, "X8_irr": -0.0690},
    # Class 4 (27.92%) - BASE
    {"X1": 0.5794, "X2": -0.9683, "X5": 1.2216, "X8_irr": -0.0635},
]


gamma0 = [
    # Class 1 (12.58%)
    {
        "Z1": 0.3929,
        "Z2": -0.9502,
        "Z3": -1.3282,
        "Z5": 1.4033,
        "Z6": 0.9985,
        "Z7": 1.5875,
        "Z8_irr": -0.1057,
        "Z9_irr": 0.0241,
    },
    # Class 2 (6.72%)
    {"Z1": -0.4306, "Z2": -1.7526, "Z3": -1.2905, "Z4": -1.3531, "Z6": 1.8623, "Z8_irr": -0.0021, "Z9_irr": 0.8050},
    # Class 3 (52.78%)
    {"Z1": 1.6298, "Z2": 0.1233, "Z3": -1.4134, "Z4": 1.1214, "Z6": -0.1236, "Z7": 1.2071, "Z8_irr": 0.0108},
    # Class 4 (27.92%) - BASE
    {"Z5": 1.3529},
]

# Class 4 as BASE
inter0 = {
    0: -0.9238,  # Inter_C1
    1: -4.1441,  # Inter_C2
    2: 1.1511,  # Inter_C3
    # Class  4 -> base, sin intercepto
}

N_INIT = 20
MAXITER = 1000


model = LatentClassFinal()
model.setup(
    X=X,
    y=y,
    varnames=ALL_VARS,
    ids=ids,
    alts=alts,
    class_params_spec=class_params_spec,
    member_params_spec=member_params_spec,
    base_class=BASE_CLASS_NEW,  # 1-indexed
    avail=av,
    ind_id=ind_ids,
    panels=True,
    betas0=betas0,
    gamma0=gamma0,
    inter0=inter0,
    n_init=N_INIT,
    maxiter=MAXITER,
    random_state=0,
    ftol=1e-15,
    gtol=1e-6,
    choice_correction=False,
    membership_correction=False,
    compute_opg=True,
    # === NUEVOS PARÁMETROS PARA CONTROLAR MEMORIA ===
    use_chunking=True,  # Cambia a False para usar el comportamiento original
    chunk_size=500,  # Tamaño del bloque de individuos procesados simultáneamente
)
model.fit()
model.summarise()
