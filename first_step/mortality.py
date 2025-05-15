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

def prob_survival(period, options, survival):
    alpha1 = options["alpha1"]
    alpha2 = options["alpha2"]

    # death probability must be jnp.exp, and clipped to [0,1]
    death_prob = alpha1 * (jnp.exp(alpha2 * period) - 1.0)
    death_prob = jnp.clip(death_prob, 0.0, 1.0)     # just in case it ever goes outside

    # alive: P(dead next) = death_prob,  P(alive next) = 1 - death_prob
    probs_alive = jnp.stack([death_prob, 1.0 - death_prob], axis=-1)   # shape (n_agents, 2)

    # dead:   P(dead next) = 1,           P(alive next) = 0
    probs_dead = jnp.broadcast_to(jnp.array([1.0, 0.0]), probs_alive.shape)

    # select rowwise based on current survival
    # survival is shape (n_agents,), so survival[...,None] is (n_agents,1)
    return jnp.where(survival[..., None] == 1, probs_alive, probs_dead)


