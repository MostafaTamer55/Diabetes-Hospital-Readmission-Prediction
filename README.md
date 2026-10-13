# Diabetes 130-US Hospitals: 30-Day Readmission Risk Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: UCI-296](https://img.shields.io/badge/Dataset-UCI%20ML%20(ID%20296)-success.svg)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
[![Domain: Healthcare ML](https://img.shields.io/badge/Domain-Healthcare%20%26%20Clinical%20ML-red.svg)](#)

An end-to-end Machine Learning pipeline predicting **30-day hospital readmission risk** for diabetic patients across 10 years of clinical care (1999–2008) at 130 US medical facilities.

---

## 🏥 Clinical Background & Problem Statement

Hospital readmission within 30 days of discharge is an essential healthcare quality indicator and a major financial liability under the **Hospital Readmissions Reduction Program (HRRP)**. Diabetic patients frequently experience complex comorbidity profiles, polypharmacy interactions, and unstable glycemic control, making early risk identification critical for targeted outpatient discharge interventions.

### Key Objectives
* Predict early 30-day inpatient readmission (`<30` days) versus late/non-readmission (`>30` or `NO`).
* Analyze the predictive significance of glycemic monitoring (HbA1c test results) and medication dosage changes on hospital readmission rates.
* Address severe clinical class imbalance (~11.1% 30-day readmissions) using cost-sensitive learning and threshold tuning.

---

## 📊 Dataset Overview (UCI ID: 296)

* **Source:** UCI Machine Learning Repository — *Diabetes 130-US hospitals for years 1999–2008*
* **Encounters:** **101,766** clinical records
* **Unique Patients:** **71,518** patients
* **Features:** **50 attributes**
  * *Demographics:* Age, Gender, Race
  * *Admission Details:* Admission Type, Discharge Disposition, Admission Source, Time in Hospital (1–14 days)
  * *Diagnostic Profiles:* Primary, secondary, and tertiary ICD-9 diagnostic codes (`diag_1`, `diag_2`, `diag_3`), Total number of diagnoses
  * *Lab & Glycemic Indicators:* Number of lab procedures, `max_glu_serum`, `A1Cresult` (HbA1c test status: None, Normal, >7, >8)
  * *Medication Regimens:* 24 specific diabetic medications tracked across dosage changes (`No`, `Steady`, `Up`, `Down`) and general medication change indicator (`change`).

---

## ⚙️ Machine Learning Pipeline Architecture

```text
 101,766 Raw Records
       │
 [ 1. Data Cleaning & Integrity ]
       ├─ Map '?' missing entries
       └─ Exclude terminal discharges (hospice / deceased)
       │
 [ 2. Clinical Feature Engineering ]
       ├─ Categorize ICD-9 diagnostic codes into 9 clinical categories (Circulatory, Diabetes, etc.)
       ├─ Binary target encoding (1: Readmitted <30 days, 0: >30 or NO)
       └─ Medication dosage adjustments & polypharmacy counts
       │
 [ 3. Class Imbalance Handling ]
       ├─ SMOTE / Random Under-Sampling
       └─ Class-weighted loss functions (Scale Pos Weight)
       │
 [ 4. Model Training & Validation ]
       ├─ Logistic Regression (Baseline)
       ├─ Random Forest
       ├─ XGBoost
       └─ LightGBM
       │
 [ 5. Clinical Performance Evaluation ]
       └─ ROC-AUC, PR-AUC, F1-Score, and High-Recall Decision Thresholds
```

---

## 📂 Repository Structure

```
Diabetes-Hospital-Readmission-Prediction/
├── README.md               # Project documentation & clinical analysis overview
├── LICENSE                 # MIT License
├── requirements.txt        # Python scientific and ML dependencies
├── data/
│   ├── diabetic_data.csv   # Complete UCI clinical dataset (101,766 records)
│   └── IDS_mapping.csv     # Categorical mapping identifiers for admissions & discharges
├── notebooks/              # Jupyter Notebooks for EDA, feature engineering, and modeling
│   └── README.md
└── src/                    # Reusable Python modules
    ├── __init__.py
    └── data_loader.py      # Automated ICD-9 grouping & clinical preprocessing pipeline
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MostafaTamer55/Diabetes-Hospital-Readmission-Prediction.git
cd Diabetes-Hospital-Readmission-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Load & Preprocess Data in Python
```python
from src.data_loader import load_raw_data, preprocess_diabetes_data

# 1. Load raw dataset
df_raw = load_raw_data()
print(f"Loaded {df_raw.shape[0]} records and {df_raw.shape[1]} attributes.")

# 2. Apply clinical preprocessing & extract 30-day binary readmission target
X, y = preprocess_diabetes_data(df_raw, binary_target=True)
print(f"Features: {X.shape[1]} | 30-Day Readmission Rate: {y.mean() * 100:.2f}%")
```

---

## 📖 Citation & Acknowledgments

* **Original Dataset:** Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records*. BioMed Research International, 2014, 781670.
* **UCI Repository:** [UCI Machine Learning Repository: Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## 📄 License
This repository is open-sourced under the MIT License - see the [LICENSE](LICENSE) file for details.
