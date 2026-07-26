import numpy as np
import pandas as pd
from sdscaler import SDScaler

df = pd.DataFrame({
    "age": [22, 25, 130, 28, 24, 26, 23, 27],           # has an outlier
    "income": [30000, 32000, 31000, 500000, 29000, 33000, 31500, 30500],  # skewed + outlier
    "score": [88, 92, 79, 85, 90, 91, 87, 89],            # normal-ish
    "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA"],  # not numeric
})

print(df)

# plain auto mode - decides per column, leaves "city" alone since it's not numeric
scaler = SDScaler()
scaled = scaler.fit_transform(df)
print(scaled)
print(scaler.summary())

# undo it
back = scaler.inverse_transform(scaled)
print(back)

# force a specific column to log-scale regardless of what auto would've picked
log_scaler = SDScaler(log_cols=["score"])
log_scaled = log_scaler.fit_transform(df)
print(log_scaler.summary())

# missing values - fit still works, just warns you
df_with_nan = df.copy()
df_with_nan.loc[0, "income"] = np.nan
nan_scaler = SDScaler()
nan_scaled = nan_scaler.fit_transform(df_with_nan)   # prints a warning about "income"
print(nan_scaled)

# building an sklearn ColumnTransformer from the same decisions, for use in a Pipeline
ct = scaler.to_column_transformer()
ct_out = ct.fit_transform(df)
print(ct_out[:3])