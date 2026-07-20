# SDScaler

Small auto-scaler for tabular data. Instead of manually picking StandardScaler
vs MinMaxScaler vs RobustScaler for each column, this just looks at the column
and picks one for you (checks for outliers and skew). You can also just force
one method if you don't trust the auto part.

Made this for a uni project, kept it simple on purpose - no sklearn
dependency, just numpy and pandas.

## Install

```
cd SDScaler
pip install -e .
```

## Usage

```python
import pandas as pd
from sdscaler import SDScaler

df = pd.DataFrame({
    "age": [22, 25, 130, 28, 24],
    "income": [30000, 32000, 31000, 500000, 29000],
})

scaler = SDScaler()
scaled = scaler.fit_transform(df)

print(scaled)
print(scaler.summary())   # which method got used per column
```

Force one method instead of auto:

```python
scaler = SDScaler(method="standard")   # or "minmax" / "robust"
```

Undo the scaling later:

```python
scaler.inverse_transform(scaled)
```

Remember to fit only on your training data, then just `.transform()` the
test set with those same stats - don't fit_transform both separately or
you'll leak info / get mismatched scales.

```python
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)
```

## How the auto part decides

For each column:
- if more than 5% of points are outliers (1.5*IQR rule) -> robust
- else if it's pretty skewed (skew > 1) -> minmax
- otherwise -> standard

Both cutoffs can be changed:

```python
SDScaler(outlier_cutoff=0.1, skew_cutoff=1.5)
```

Heads up - on small datasets one weird value can be enough to trip the
outlier check even if the column mostly looks normal, so don't be surprised
if a small demo dataset gets more "robust" columns than you'd expect.

## Files

```
sdscaler/
  scaler.py    - the SDScaler class
  helpers.py   - skew/outlier detection logic
examples/
  example_usage.py
tests/
  test_scaler.py
```

## Running tests

```
pip install pytest
pytest tests/
```

## Todo / ideas

- log transform option
- skip non-numeric columns automatically instead of erroring
- maybe a plot function to compare before/after
