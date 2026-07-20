import pandas as pd
from sdscaler import SDScaler

df = pd.DataFrame({
    "age": [22, 25, 130, 28, 24, 26, 23, 27],
    "income": [30000, 32000, 31000, 500000, 29000, 33000, 31500, 30500],
    "score": [88, 92, 79, 85, 90, 91, 87, 89],
})

print(df)

scaler = SDScaler()
scaled = scaler.fit_transform(df)
print(scaled)
print(scaler.summary())

back = scaler.inverse_transform(scaled)
print(back)
