import numpy as np
import matplotlib.pyplot as plt

# Tax parameters
th1 = 48_000     # First threshold in DKK
th2 = 569_800    # Second threshold in DKK
r2  = 0.38       # Tax rate between th1 and th2
r3  = 0.50       # Tax rate above th2

# 1) Create a grid of incomes from 0 to 750,000 DKK
income = np.linspace(0, 750_000, 500)

# 2) Compute the portions in each bracket
inc1 = np.minimum(income, th1)
inc2 = np.minimum(np.maximum(income - th1, 0.0), th2 - th1)
inc3 = np.maximum(income - th2, 0.0)

# 3) Compute total tax liability
tax = r2 * inc2 + r3 * inc3

# 4) Plot the tax schedule
plt.figure(figsize=(8, 5))
plt.plot(income, tax, linewidth=2, label="Tax Liability")
plt.axvline(th1, color='gray', linestyle='--', label='Threshold 1 (48.000 DKK)')
plt.axvline(th2, color='gray', linestyle=':', label='Threshold 2 (569.800 DKK)')
plt.xlabel("Labor Income (DKK)")
plt.ylabel("Tax Liability (DKK)")
plt.title("Piecewise Linear Tax Function")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()