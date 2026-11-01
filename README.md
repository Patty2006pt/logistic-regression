# logistic-regression

# Breast Cancer Prediction with Multiple Machine Learning Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A data science project utilizing Python, Scikit-Learn, and Pandas to load, preprocess, scale, and analyze breast cancer dataset features for binary classification tasks.

## 🚀 Features
- 📊 Loads and explores the benchmark Breast Cancer Wisconsin dataset from `scikit-learn`[cite: 4].
- 🗂️ Preprocesses data into feature matrices ($X$) and target vectors ($y$), converting target classes into binary representations[cite: 4].
- ⚖️ Applies feature scaling using `StandardScaler` to normalize input features for machine learning workflows[cite: 4].
- 📈 Visualizes multi-variable feature distributions and relationships using Seaborn pair plots[cite: 4].

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:**
  - `pandas` (Data manipulation and structuring)[cite: 4]
  - `scikit-learn` (Dataset loading and data preprocessing)[cite: 4]
  - `seaborn` / `matplotlib` (Data visualization)[cite: 4]

## ⚙️ Getting Started

### Prerequisites
Make sure you have Python installed along with the required analytical packages:
```bash
pip install pandas scikit-learn seaborn matplotlib

Installation & Execution
1. Clone this repository or copy the code snippet into your local Jupyter Notebook or Google Colab environment.
2. Run the preprocessing script
python cancer_analysis.py

📖 Usage
Here is a core snippet demonstrating how the dataset is loaded, mapped, and scaled:
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load dataset
cancer = datasets.load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.DataFrame(cancer.target, columns=['cancer'])

# Map target labels to binary format (1 = malignant, 0 = benign)
y['cancer'] = y.cancer.apply(lambda x: 1 if x == 0 else 0)
df = pd.concat([X, y], axis=1)

# Standardize features
scaler = StandardScaler()
X_sca = scaler.fit_transform(X)
X_sca = pd.DataFrame(X_sca, columns=cancer.feature_names)
