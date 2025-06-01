import pandas as pd
import matplotlib.pyplot as plt

# 1) Indlæs data fra Excel
df = pd.read_excel('data/jobindsats.xlsx')

# 2) Konverter eventuelle decimalkommaer til punktum og til float
df['employment_rate'] = df['employment_rate'].astype(str).str.replace(',', '.').astype(float)

# 3) Pivot så vi får år som kolonner
df_pivot = df.pivot(index='age', columns='year', values='employment_rate')

# 4) Beregn forskellen 2025 - 2023
df_pivot['diff_2025_2023'] = df_pivot[2025] - df_pivot[2023]

# 5) Plot differencen
plt.figure(figsize=(8, 5))
df_pivot['diff_2025_2023'].plot(kind='bar')
plt.xlabel('Alder')
plt.ylabel('Ændring i beskæftigelsesrate (2025 – 2023)')
plt.title('Forskel i beskæftigelsesrate efter alder: 2025 vs 2023')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 6) Gem figuren (valgfrit)
plt.savefig('diff_2025_2023_by_age.png', dpi=150, bbox_inches='tight')

plt.show()