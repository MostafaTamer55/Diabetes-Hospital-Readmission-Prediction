# -*- coding: utf-8 -*-
"""
src.data_loader
===============
Data loading, cleaning, and clinical feature engineering utilities for the
UCI Diabetes 130-US Hospitals (1999-2008) dataset.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple


def get_data_path() -> str:
    """Returns absolute path to diabetic_data.csv."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', 'diabetic_data.csv')


def load_raw_data() -> pd.DataFrame:
    """
    Loads raw CSV and replaces '?' with NaN.
    """
    path = get_data_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    df = pd.read_csv(path, na_values=['?'], low_memory=False)
    return df


def categorize_icd9(code: str) -> str:
    """
    Maps ICD-9 diagnostic codes into primary clinical diagnostic categories.
    Based on standard healthcare data mining literature (Strack et al., 2014).
    """
    if pd.isna(code):
        return 'Missing'
    code_str = str(code).strip()
    
    # Check E and V codes
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
        
    try:
        val = float(code_str)
    except ValueError:
        return 'Other'
        
    if (390 <= val <= 459) or val == 785:
        return 'Circulatory'
    elif (460 <= val <= 519) or val == 786:
        return 'Respiratory'
    elif (520 <= val <= 579) or val == 787:
        return 'Digestive'
    elif 250 <= val < 251:
        return 'Diabetes'
    elif (800 <= val <= 999):
        return 'Injury'
    elif (710 <= val <= 739):
        return 'Musculoskeletal'
    elif (580 <= val <= 629) or val == 788:
        return 'Genitourinary'
    elif (140 <= val <= 239):
        return 'Neoplasms'
    else:
        return 'Other'


def preprocess_diabetes_data(df: pd.DataFrame, binary_target: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Performs standard clinical preprocessing:
      - Drops encounter_id and patient_nbr identifiers.
      - Drops columns with >50% missing values (weight, payer_code, medical_specialty).
      - Converts diagnosis codes (diag_1, diag_2, diag_3) into ICD-9 categories.
      - Encodes target variable:
          If binary_target=True: 1 for readmitted <30 days, 0 otherwise (NO or >30).
    """
    data = df.copy()

    # Drop high-missing rate columns
    cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Remove rows where patient died or was discharged to hospice (dispositions 11, 19, 20, 21)
    if 'discharge_disposition_id' in data.columns:
        data = data[~data['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    # Group ICD-9 diagnosis
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in data.columns:
            data[f'{col}_group'] = data[col].apply(categorize_icd9)
            data = data.drop(columns=[col])

    # Prepare Target Variable
    if 'readmitted' in data.columns:
        if binary_target:
            # 1 = Readmitted within 30 days (<30), 0 = >30 or NO
            y = (data['readmitted'] == '<30').astype(int)
        else:
            y = data['readmitted']
        X = data.drop(columns=['readmitted'])
    else:
        y = None
        X = data

    return X, y
