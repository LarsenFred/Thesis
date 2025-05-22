#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ——————————————————————————————————————————————
# 1) locate & load spline parameters
PARAM_FILE = "/Users/frederiklarsen/Thesis/Thesis/first_step/oap_params.txt"

b0, b1, b2, b3 = np.loadtxt(PARAM_FILE)

# ——————————————————————————————————————————————
# 2) define thresholds and grid
supp_threshold = 79_300   # k1
oap_threshold  = 335_920  # k2
income         = np.linspace(0, 600_000, 500)

# ——————————————————————————————————————————————
# 3) true policy schedule
supp_reduction_rate = 0.309
income_over_supp    = np.maximum(0, income - supp_threshold)
supplement = np.maximum(0, 92_940 - supp_reduction_rate * income_over_supp)

oap_reduction_rate = 0.30
income_over_oap     = np.maximum(0, income - oap_threshold)
base_pension = np.maximum(0, 80_328 - oap_reduction_rate * income_over_oap)

annual_pension = base_pension + supplement

# ——————————————————————————————————————————————
# 4) spline‐based prediction
def predict_oap(income_array):
    inc = np.asarray(income_array)
    L1  = np.maximum(0, inc - supp_threshold)
    L2  = np.maximum(0, inc - oap_threshold)
    return b0*100000 + b1*inc + b2*L1 + b3*L2

predicted_pension = predict_oap(income)

# ——————————————————————————————————————————————
# 5) plot
plt.figure(figsize=(8,5))
plt.plot(income, annual_pension,    lw=2, label="True Old Age Pension")
plt.plot(income, predicted_pension, lw=2, linestyle="--", label="Estimated Old Age Pension")

# vertical lines at kinks
plt.axvline(supp_threshold, color="green", linestyle=":", label="Supplement Threshold")
plt.axvline(oap_threshold,  color="purple", linestyle=":", label="Pension Threshold")

plt.xlabel("Annual Labor Income (DKK)")
plt.ylabel("Annual Old‐Age Pension (DKK)")
plt.title("True vs. Estimated Pension Schedule")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/frederiklarsen/Library/Mobile Documents/com~apple~CloudDocs/KU/Speciale/figurer/oap_spline_estimate.png", dpi=300)
plt.show()
