import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Tax parameters
th1 = 48_000     # First threshold in DKK
th2 = 569_800    # Second threshold in DKK
r2  = 0.38       # Tax rate between th1 and th2
r3  = 0.50       # Tax rate above th2

# Create a grid of incomes from 0 to 1,500,000 DKK
income = np.linspace(0, 1_500_000, 500)

# Compute the portions in each bracket
inc2 = np.minimum(np.maximum(income - th1, 0.0), th2 - th1)
inc3 = np.maximum(income - th2, 0.0)

# Compute total tax liability
tax = r2 * inc2 + r3 * inc3

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(income, tax, linewidth=2, label="Tax Liability")
ax.axvline(th1, color='gray', linestyle='--', label='Base Threshold (48.000 DKK)')
ax.axvline(th2, color='gray', linestyle=':', label='Top Threshold (569.800 DKK)')
ax.set_xlabel("Labor Income (DKK)")
ax.set_ylabel("Tax Liability (DKK)")
ax.set_title("Piecewise Linear Tax Function")
ax.grid(True)
ax.legend()

# Disable scientific notation on both axes
fmt = ScalarFormatter(useOffset=False)
fmt.set_scientific(False)
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)

plt.tight_layout()
plt.show()