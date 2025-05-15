# pension_spline.py

import numpy as np
import statsmodels.api as sm

import sys
sys.path.insert(0,"/Users/frederiklarsen/dcegm/Speciale")


# estimate old age pension as function of income
# k1 = options["model_params"]["supp_threshold"]        # e.g. 0.793 → 79 300 kr
# k2 = options["model_params"]["oap_threshold"]         # e.g. 3.3592 → 335 920 kr
# max_supp = options["model_params"]["oap_max_supplement"]
# supp_rate = options["model_params"]["supp_reduction_rate"]
# base_oap  = options["model_params"]["oap_base_amount"]
# oap_rate  = options["model_params"]["oap_reduction_rate"]

base_oap = 0.80328,
max_supp = 0.92940,
k1 = 0.79300,
k2 = 3.3592,
supp_rate = 0.309,
oap_rate = 0.3,

# 2) Build a grid of (relative) incomes from 0 to 6 (=600 000 kr)
inc = np.linspace(0, 6, 500)  

# 3) Compute the schedule exactly as you did before
inc_over_supp = np.maximum(0, inc - k1)
supplement    = np.maximum(0, max_supp - supp_rate * inc_over_supp)

inc_over_oap  = np.maximum(0, inc - k2)
oap           = np.maximum(0, base_oap - oap_rate * inc_over_oap)

annual_pension = supplement + oap  # in “per 100 000 kr” units

# 4) Create the linear‐spline basis (two kinks at k1 and k2)
L1 = np.maximum(0, inc - k1)
L2 = np.maximum(0, inc - k2)

# 5) Stack into a design matrix [1, inc, (inc−k1)+, (inc−k2)+]
X = np.column_stack([inc, L1, L2])
X = sm.add_constant(X)  # adds an intercept β₀

# 6) Run the regression
spline_model = sm.OLS(annual_pension, X).fit()

print(spline_model.summary())

# 7) Extract the four coefficients
β0, β1, β2, β3 = spline_model.params
print(f"β₀ = {β0:.6f}, β₁ = {β1:.6f}, β₂ = {β2:.6f}, β₃ = {β3:.6f}")

# save as txt file
np.savetxt("/Users/frederiklarsen/dcegm/Speciale/first_step/oap_params.txt", [β0, β1, β2, β3])
