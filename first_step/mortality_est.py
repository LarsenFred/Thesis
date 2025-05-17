import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize

DATA_DIR    = "/Users/frederiklarsen/Data"
Mortality   = os.path.join(DATA_DIR, "mortality.xlsx")


df = pd.read_excel(Mortality, sheet_name="DOD")
ages  = df["age"].values     # e.g. 35,36,…,85
q_obs = df["mortality"].values  # observed probability of dying at each age

# --- 2) define the model hazard
def hazard(ages, alpha1, alpha2, a0=30):
    return alpha1 * (np.exp(alpha2 * (ages - a0)) - 1)

# --- 3) negative log‐likelihood
def neg_loglik(theta, ages, q_obs, a0=30):
    alpha1, alpha2 = theta
    p = hazard(ages, alpha1, alpha2, a0=a0)
    p = np.clip(p, 1e-8, 1-1e-8)
    ll = q_obs * np.log(p) + (1 - q_obs) * np.log(1 - p)
    return -np.sum(ll)

# --- 4) MLE
init = np.array([0.0005, 0.1])
bnds = [(1e-10, 1.0), (1e-5, 1.0)]
res = minimize(
    neg_loglik,
    init,
    args=(ages, q_obs),
    method="L-BFGS-B",
    bounds=bnds,
    options={"disp": True}
)

alpha1_hat, alpha2_hat = res.x
print(f"MLE estimates: alpha1 = {alpha1_hat:.6f}, alpha2 = {alpha2_hat:.6f}")

#print as txt file
np.savetxt("/Users/frederiklarsen/dcegm/Speciale/first_step/mortality_params.txt", [alpha1_hat, alpha2_hat])

#plot predicted vs actual
# plt.figure(figsize=(5, 3))
# plt.plot(ages, q_obs, label="Observed Mortality", marker='o')
# plt.plot(ages, hazard(ages, alpha1_hat, alpha2_hat), label="Predicted Mortality", linestyle='--')
# plt.xlabel("Age")
# plt.ylabel("Mortality Rate")
# plt.title("Predicted vs Actual Mortality Rate")
# plt.legend()
# plt.grid()
# plt.show()
