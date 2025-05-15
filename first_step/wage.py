import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_DIR    = "/Users/frederiklarsen/Data"
FILE_UDD1   = os.path.join(DATA_DIR, "momenter/moments_udd1.txt")
FILE_UDD2   = os.path.join(DATA_DIR, "momenter/moments_udd2.txt")
FILE_UDD3   = os.path.join(DATA_DIR, "momenter/moments_udd3.txt")


#=================================
#       Education level 1
#=================================

df_edu_1 = pd.read_csv(FILE_UDD1)
df_edu_2 = pd.read_csv(FILE_UDD2)
df_edu_3 = pd.read_csv(FILE_UDD3)

# 2) standardize colomn name and remove _FREQ_ column
for df in (df_edu_1, df_edu_2, df_edu_3):
    # Rename ALDER → age
    if "ALDER" in df.columns:
        df.rename(columns={"ALDER": "age"}, inplace=True)
    # Remove _FREQ_-column if it exists
    if "_FREQ_" in df.columns:
        df.drop(columns=["_FREQ_"], inplace=True)


# 2) Build the regression dataset
df_edu_1['log_wage'] = np.log(df_edu_1['avg_wage'])
X = sm.add_constant(np.column_stack((df_edu_1['age'], df_edu_1['age']**2)))   # [1, age, age²] 

# 3) Run OLS: log(wage) = β0 + β1·age + β2·age² + error
model1 = sm.OLS(df_edu_1['log_wage'], X, missing='drop').fit()

# 4) Extract your β’s
beta0_1, beta1_1, beta2_1 = model1.params
print(f"Estimated betas: β0={beta0_1:.4f}, β1={beta1_1:.4f}, β2={beta2_1:.4f}")

# save as txt file
np.savetxt("/Users/frederiklarsen/dcegm/Speciale/first_step/wage_params_udd1.txt", [beta0_1, beta1_1, beta2_1])

df_edu_1["predicted"]=np.exp(beta0_1 + beta1_1*df_edu_1["age"] + beta2_1*df_edu_1["age"]**2)


#=================================
#       Education level 2
#=================================

# 2) Build the regression dataset
df_edu_2['log_wage'] = np.log(df_edu_2['avg_wage'])
X = sm.add_constant(np.column_stack((df_edu_2['age'], df_edu_2['age']**2)))   # [1, age, age²] 

# 3) Run OLS: log(wage) = β0 + β1·age + β2·age² + error
model2 = sm.OLS(df_edu_2['log_wage'], X, missing='drop').fit()

# 4) Extract your β’s
beta0_2, beta1_2, beta2_2 = model2.params
print(f"Estimated betas: β0={beta0_2:.4f}, β1={beta1_2:.4f}, β2={beta2_2:.4f}")

# save as txt file
np.savetxt("/Users/frederiklarsen/dcegm/Speciale/first_step/wage_params_udd2.txt", [beta0_2, beta1_2, beta2_2])

df_edu_1["predicted"]=np.exp(beta0_2 + beta1_2*df_edu_1["age"] + beta2_2*df_edu_1["age"]**2)


#=================================
#       Education level 3
#=================================

# 2) Build the regression dataset
df_edu_3['log_wage'] = np.log(df_edu_3['avg_wage'])
X = sm.add_constant(np.column_stack((df_edu_3['age'], df_edu_3['age']**2)))   # [1, age, age²] 

# 3) Run OLS: log(wage) = β0 + β1·age + β2·age² + error
model3 = sm.OLS(df_edu_3['log_wage'], X, missing='drop').fit()

# 4) Extract your β’s
beta0_3, beta1_3, beta2_3 = model3.params
print(f"Estimated betas: β0={beta0_3:.4f}, β1={beta1_3:.4f}, β2={beta2_3:.4f}")

# save as txt file
np.savetxt("/Users/frederiklarsen/dcegm/Speciale/first_step/wage_params_udd3.txt", [beta0_3, beta1_3, beta2_3])

df_edu_1["predicted"]=np.exp(beta0_3 + beta1_3*df_edu_1["age"] + beta2_3*df_edu_1["age"]**2)

