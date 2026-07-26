# SDScaler

Small auto-scaler for tabular data. Instead of manually picking StandardScaler
vs MinMaxScaler vs RobustScaler for each column, this just looks at the column
and picks one for you (checks for outliers and skew). You can also just force
one method if you don't trust the auto part.

Made this for a uni project, kept it simple on purpose - no sklearn
dependency for the core logic (sklearn's only needed if you use
`to_column_transformer()`), just numpy and pandas.

## Install

\```
cd SDScaler
pip install -e .
\```

## Usage

\```python
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
\```

Force one method instead of auto:

\```python
scaler = SDScaler(method="standard")   # or "minmax" / "robust" / "log"
\```

Undo the scaling later:

\```python
scaler.inverse_transform(scaled)
\```

Remember to fit only on your training data, then just `.transform()` the
test set with those same stats - don't fit_transform both separately or
you'll leak info / get mismatched scales.

\```python
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)
\```

## How the auto part decides

For each numeric column:
- if more than 5% of points are outliers (1.5*IQR rule) -> robust
- else if it's pretty skewed (skew > 1) and strictly positive -> log
- else if it's pretty skewed but has zero/negative values -> minmax
- otherwise -> standard

Both cutoffs can be changed:

\```python
SDScaler(outlier_cutoff=0.1, skew_cutoff=1.5)
\```

Heads up - on small datasets one weird value can be enough to trip the
outlier check even if the column mostly looks normal, so don't be surprised
if a small demo dataset gets more "robust" columns than you'd expect.

## Forcing log scaling on specific columns

\```python
scaler = SDScaler(log_cols=["income"])
\```

This scales `income` with log regardless of what auto-pick would've chosen,
everything else still gets decided normally. Log scaling shifts the column
to be positive first if it isn't already, so it won't blow up on zeros or
negative values, but it's really meant for stuff that's naturally always
positive and skewed (income, population, prices, that kind of thing).

## Non-numeric columns

If you pass a dataframe that still has a categorical/text column in it (like
forgot to encode it, or it's a column you're not scaling on purpose), SDScaler
just leaves it alone instead of erroring out. Only numeric columns get touched.

\```python
scaler.summary()
#   column   method
# 0    age   standard
# 1   city   passthrough (not numeric)
\```

## Missing values

If a column has NaN in it, `.fit()` still works and warns you about which
columns have missing data. The stats (mean, median, etc.) get computed
ignoring the NaNs, but the scaled output will still have NaN in those exact
spots - didn't want to silently fill anything in without you knowing.

## Using it with sklearn Pipelines

\```python
scaler = SDScaler().fit(df)
ct = scaler.to_column_transformer()
\```

Builds an sklearn `ColumnTransformer` using the same decisions SDScaler
already made, for anyone who'd rather have it inside a Pipeline instead of
calling `.transform()` directly. Non-numeric columns get passed through here
too. Needs scikit-learn installed, only used if you actually call this.

## Files

\```
sdscaler/
  scaler.py    - the SDScaler class
  helpers.py   - skew/outlier detection logic
examples/
  example_usage.py
tests/
  test_scaler.py
\```

## Running tests

\```
pip install pytest
pytest tests/
\```

## Todo / ideas

- frequency encoding style option for very high-cardinality numeric IDs
- maybe a plot function to compare before/after distributions
- log scaling on negative-heavy data currently just shifts everything, could
  be smarter about it for columns that are mostly negative