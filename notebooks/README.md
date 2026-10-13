# Notebooks Directory

Place your Jupyter Notebooks here (e.g. `diabetes_readmission_pipeline.ipynb` or `model_training.ipynb`).

You can easily load the preprocessed data using the provided pipeline:

```python
import sys
sys.path.append('..')

from src.data_loader import load_raw_data, preprocess_diabetes_data

# Load raw dataset
raw_df = load_raw_data()
print("Raw shape:", raw_df.shape)

# Clean, group ICD-9 codes, and extract binary 30-day readmission target
X, y = preprocess_diabetes_data(raw_df, binary_target=True)
print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts(normalize=True))
```
