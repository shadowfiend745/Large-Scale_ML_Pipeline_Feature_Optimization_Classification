# End_to_end_ML_Pipeline_Feature_Optimization_Classification

A machine learning pipeline that classifies obesity levels using K-Nearest Neighbours (KNN), with support for dimensionality reduction via **PCA** or feature selection via **RFE (Recursive Feature Elimination)**.

---

## Dataset

**`ObesityDataSet_raw_and_data_sinthetic.csv`**

Contains raw and synthetically augmented records of individuals with various lifestyle and physical attributes. The target variable is `NObeyesdad` (obesity level), a multi-class label.

---

## Project Structure

```
.
├── ObesityDataSet_raw_and_data_sinthetic.csv   # Input dataset
├── Compare_feature_selection_on_classifier.py  # Main script
└── utilities.py                                # Helper functions
```

---

## Pipeline Overview

### 1. Preprocessing (`Compare_feature_selection_on_classifier.py`)

- Binary columns (`family_history_with_overweight`, `FAVC`, `SMOKE`, `SCC`) are mapped to `1`/`0`
- `Gender` is encoded as `Male=1`, `Female=0`
- Ordinal columns (`CAEC`, `CALC`) are mapped to frequency levels: `no=0`, `Sometimes=1`, `Frequently=2`, `Always=3`
- `MTRANS` (transport mode) is one-hot encoded via `pd.get_dummies`
- Target variable `NObeyesdad` is label-encoded

### 2. Data Splitting

The dataset is split into three sets:

| Split | Size |
|-------|------|
| Train | 60%  |
| Validation | 20% |
| Test | 20%  |

### 3. Standardisation

All features are z-score standardised using training set statistics (mean and std) to prevent data leakage. Test and validation sets are transformed using the training set's parameters.

### 4. Feature Selection / Dimensionality Reduction

Two modes are available (toggled via boolean flags at the top of the main script):

#### PCA Mode (`pcaMode = True`)
- Custom PCA implementation using covariance matrix and eigenvector decomposition
- Reduces features to `nComponents` principal components
- Eigenvectors are fitted on training data and applied to val/test sets

#### RFE Mode (`rfeMode = True`)
- Uses scikit-learn's `RFE` with `LogisticRegression` as the estimator
- Selects the top 10 most informative features
- Transform is fitted on training data and applied to val/test sets

### 5. K Selection

An automated k-selection function evaluates a logarithmically-spaced range of odd k values (from 1 to √n) against the validation set, and selects the k with the highest accuracy.

### 6. KNN Classification

Custom KNN implementation using Euclidean distance. Predictions are made on the test set using the best k found in the previous step.

### 7. Evaluation

A confusion matrix is generated and visualised as a heatmap using `seaborn`, with actual vs predicted class labels.

---

## Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run

```bash
python Compare_feature_selection_on_classifier.py
```

### Toggle Modes

At the top of `Compare_feature_selection_on_classifier.py`, set the desired mode:

```python
pcaMode = False   # Set to True to use PCA
rfeMode = True    # Set to True to use RFE
```

> Note: Both flags can be set independently. If both are `True`, both pipelines run and the final confusion matrix reflects the RFE predictions.

---

## Utilities (`utilities.py`)

| Function | Description |
|----------|-------------|
| `standarizationCal` | Z-score standardisation for train/test modes |
| `pcaCal` | Custom PCA using eigenvector decomposition |
| `euclideanDisCal` | Euclidean distance between two vectors |
| `knnCal` | KNN prediction using Euclidean distance |
| `kSelection` | Automated odd-k selection via validation accuracy |
| `confusionMatrixGen` | Builds a confusion matrix from true/predicted labels |
| `visualizeCM` | Heatmap visualisation of the confusion matrix |
| `csvConverter` | Converts semicolon-delimited CSV to comma-delimited |

---

## Notes

- All custom implementations (standardisation, PCA, KNN, confusion matrix) are built from scratch using NumPy — no scikit-learn wrappers are used for the core ML logic.
- RFE uses scikit-learn's `LogisticRegression` solely as the feature ranking estimator, not as the final classifier.
- The KNN implementation may be slow on large datasets due to brute-force distance computation.

