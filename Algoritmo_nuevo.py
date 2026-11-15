"""
Benchmark Comparativo: Algoritmo Original vs. Optimizado (Chunking / lax.scan)
============================================================================
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

BASE_CLASS_NEW = 3  # 1-indexed

# =====================================================================
# WARM START FROM SYN_12
# =====================================================================
betas0 = [
    {"X2": 1.1921, "X3": -0.0262, "X4": -1.3334, "X6": -0.9892, "X7_irr": -0.1569, "X8_irr": -0.1815},
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
    {"X1": -0.9570, "X2": 0.7446, "X3": 1.4801, "X4": -0.0109, "X6": 0.8479, "X8_irr": -0.0690},
    {"X1": 0.5794, "X2": -0.9683, "X5": 1.2216, "X8_irr": -0.0635},
]

gamma0 = [
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
    {"Z1": -0.4306, "Z2": -1.7526, "Z3": -1.2905, "Z4": -1.3531, "Z6": 1.8623, "Z8_irr": -0.0021, "Z9_irr": 0.8050},
    {"Z1": 1.6298, "Z2": 0.1233, "Z3": -1.4134, "Z4": 1.1214, "Z6": -0.1236, "Z7": 1.2071, "Z8_irr": 0.0108},
    {"Z5": 1.3529},
]

inter0 = {
    0: -0.9238,
    1: -4.1441,
    2: 1.1511,
}

N_INIT = 20
MAXITER = 1000

# =====================================================================
# 1. EJECUCIÓN MÉTODO ORIGINAL
# =====================================================================
print("=" * 60)
print("INICIANDO BENCHMARK: MÉTODO ORIGINAL (Vectorizado Completo)")
print("=" * 60)
model_orig = LatentClassFinal()
t_start_orig = time.time()
model_orig.setup(
    X=X, y=y, varnames=ALL_VARS, ids=ids, alts=alts,
    class_params_spec=class_params_spec,
    member_params_spec=member_params_spec,
    base_class=BASE_CLASS_NEW,
    avail=av, ind_id=ind_ids, panels=True,
    betas0=betas0, gamma0=gamma0, inter0=inter0,
    n_init=N_INIT, maxiter=MAXITER, random_state=0,
    ftol=1e-15, gtol=1e-6,
    choice_correction=False, membership_correction=False,
    compute_opg=True,
    use_chunking=False  # Desactivado
)
model_orig.fit()
t_end_orig = time.time()
time_orig = t_end_orig - t_start_orig
ll_orig = model_orig.loglik
iter_orig = model_orig.total_iter

# =====================================================================
# 2. EJECUCIÓN MÉTODO OPTIMIZADO (Chunking / lax.scan)
# =====================================================================
print("\n" + "=" * 60)
print("INICIANDO BENCHMARK: MÉTODO OPTIMIZADO (Chunking + lax.scan)")
print("=" * 60)
model_opt = LatentClassFinal()
t_start_opt = time.time()
model_opt.setup(
    X=X, y=y, varnames=ALL_VARS, ids=ids, alts=alts,
    class_params_spec=class_params_spec,
    member_params_spec=member_params_spec,
    base_class=BASE_CLASS_NEW,
    avail=av, ind_id=ind_ids, panels=True,
    betas0=betas0, gamma0=gamma0, inter0=inter0,
    n_init=N_INIT, maxiter=MAXITER, random_state=0,
    ftol=1e-15, gtol=1e-6,
    choice_correction=False, membership_correction=False,
    compute_opg=True,
    use_chunking=True,   # Activado
    chunk_size=500       # Tamaño de bloque en VRAM
)
model_opt.fit()
t_end_opt = time.time()
time_opt = t_end_opt - t_start_opt
ll_opt = model_opt.loglik
iter_opt = model_opt.total_iter

# =====================================================================
# 3. AUDITORÍA NUMÉRICA GRANULAR Y TABLA RESUMEN
# =====================================================================
def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

diff_ll = abs(ll_orig - ll_opt)
diff_betas = max(max_abs_diff(b1, b2) for b1, b2 in zip(model_orig.class_betas, model_opt.class_betas))
diff_gammas = max_abs_diff(model_orig.gamma_params, model_opt.gamma_params)
diff_se = max_abs_diff(model_orig.stderr, model_opt.stderr)
diff_post = max_abs_diff(model_orig.posterior, model_opt.posterior)
diff_probs = max_abs_diff(model_orig.class_probs, model_opt.class_probs)

print("\n" + "=" * 88)
print("                    RESUMEN COMPARATIVO DE RENDIMIENTO Y RESULTADOS")
print("=" * 88)
print(f"{'Métrica / Indicador':<36} | {'Original':<18} | {'Optimizado':<18} | {'Diff Máx (Abs)'}")
print("-" * 88)
print(f"{'Log-Likelihood Final (LL)':<36} | {ll_orig:<18.10f} | {ll_opt:<18.10f} | {diff_ll:.2e}")
print(f"{'Tiempo de Ejecución (s)':<36} | {time_orig:<18.2f} | {time_opt:<18.2f} | {time_opt - time_orig:+.2f}s")
print(f"{'Iteraciones Totales EM':<36} | {iter_orig:<18} | {iter_opt:<18} | {iter_opt - iter_orig}")
print(f"{'Coeficientes de Elección (Betas)':<36} | {'--':<18} | {'--':<18} | {diff_betas:.2e}")
print(f"{'Coeficientes Membresía (Gammas)':<36} | {'--':<18} | {'--':<18} | {diff_gammas:.2e}")
print(f"{'Errores Estándar (Hessiano)':<36} | {'--':<18} | {'--':<18} | {diff_se:.2e}")
print(f"{'Proporción de Clases (Shares)':<36} | {'--':<18} | {'--':<18} | {diff_probs:.2e}")
print(f"{'Probabilidades Posteriores (R_ic)':<36} | {'--':<18} | {'--':<18} | {diff_post:.2e}")
print("=" * 88)

# Desglose parámetro por parámetro (Betas denormalizados)
print("\nDESGLOSE DETALLADO POR PARÁMETRO (CLASE 1 A 3):")
print(f"{'Parámetro':<20} | {'Original':<16} | {'Optimizado':<16} | {'Diff Absoluta'}")
print("-" * 74)
for name, v_orig, v_opt in zip(model_orig.coeff_names, model_orig.coeff_est, model_opt.coeff_est):
    print(f"{name:<20} | {v_orig:<16.10f} | {v_opt:<16.10f} | {abs(v_orig - v_opt):.2e}")
print("=" * 74)