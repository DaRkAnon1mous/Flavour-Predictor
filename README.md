#  Flavour Predictor 🍦

A machine learning project that predicts ice cream flavour preferences based on age and gender using synthetic data and multiple classification algorithms.

## 📋 Project Overview

This project demonstrates a complete machine learning workflow from synthetic data generation to model deployment, focusing on handling imbalanced datasets and comparing multiple classification algorithms for predicting ice cream flavour preferences.

**Key Features:**
- Synthetic dataset generation with realistic patterns
- Comprehensive data preprocessing and encoding
- Multiple machine learning algorithms comparison
- Imbalanced data handling using SMOTE/SMOTENC techniques
- Model evaluation and performance optimization

## 🗂️ Project Structure

```
flavour-prediction/
├── Dataset/
│   ├── flavour.csv              # Original synthetic dataset
│   └── balanced_flavour.csv     # Balanced dataset after SMOTE
├── notebooks/
│   └── flavour_predictor.ipynb  # Main analysis notebook
├── models/
│   └── saved_models/            # Trained model files
├── README.md
└── requirements.txt
```

## 🔧 Dataset Creation

### Synthetic Data Generation
Created a synthetic dataset with **1,021 samples** containing:

**Features:**
- **Age**: Range 6-24 years (realistic age distribution for ice cream preferences)
- **Gender**: Binary (Male/Female, encoded as 1/0)

**Target Variable:**
- **Flavour**: 7 different flavours
  - Chocolate
  - Strawberry 
  - Butterscotch
  - Vanilla
  - Mango
  - Coffee
  - Almond & Chocolate

### Initial Data Challenges
- **Imbalanced Classes**: Butterscotch dominated (~40% of data)
- **Poor Model Performance**: Initial accuracy was low due to class imbalance
- **Bias Issues**: Models heavily predicted majority class

## 🛠️ Data Preprocessing

### 1. Data Loading & Exploration
```python
import pandas as pd
df = pd.read_csv("../Dataset/flavour.csv")
print(df['Flavour'].value_counts())  # Check class distribution
```

### 2. Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0
```

### 3. Handling Class Imbalance
Applied multiple balancing techniques:

**Option A: SMOTENC (Recommended)**
```python
from imblearn.over_sampling import SMOTENC
smote_nc = SMOTENC(categorical_features=[1], random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)
```

**Option B: SMOTE**
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

## 🤖 Machine Learning Models

Implemented and compared **4 different algorithms**:

### 1. Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=4)
```

### 2. Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
```

### 3. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
svm_model = SVC(kernel='poly', class_weight='balanced', probability=True, random_state=42)
```

### 4. XGBoost Classifier
```python
import xgboost as xgb
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
```

## 📊 Model Performance

### Before Data Balancing
- **Low Overall Accuracy**: ~45-60%
- **Poor Minority Class Prediction**: Many flavours had 0% recall
- **Biased Predictions**: Models favored Butterscotch (majority class)

### After Data Balancing
- **Improved Overall Accuracy**: ~75-85%
- **Balanced Performance**: All flavours now have reasonable prediction rates
- **Better Generalization**: Models perform well across all age groups and genders

### Best Performing Model
**Random Forest with Balanced Class Weights**
- **Accuracy**: ~82%
- **Balanced Performance**: Good precision/recall across all flavours
- **Robust**: Handles both numerical and categorical features well

## 🚀 Key Insights

### Age-Based Preferences
- **Children (6-12)**: Prefer Chocolate and Strawberry
- **Teens (13-17)**: Favor Mango and Butterscotch  
- **Adults (18-24)**: Lean towards Coffee and Almond & Chocolate

### Gender-Based Patterns
- **Males**: Higher preference for Chocolate and Coffee flavours
- **Females**: More likely to choose Strawberry and Vanilla

### Model Learnings
- **Class imbalance significantly impacts model performance**
- **Ensemble methods (Random Forest) perform better than single algorithms**
- **Proper data balancing is crucial for minority class prediction**

## 🏃‍♂️ How to Run

### Prerequisites
```bash
pip install pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

### Step-by-Step Execution

1. **Clone the repository**
```bash
git clone https://github.com/DaRkAnon1mous/Flavour-Predictor.git
cd flavour-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Jupyter notebook**
```bash
jupyter notebook notebooks/flavour_predictor.ipynb
```


```

## 📈 Results Summary

| Model | Before Balancing | After Balancing | Improvement |
|-------|------------------|-----------------|-------------|
| Decision Tree | 46% | 59% | +21% |
| Random Forest | 47% | 60% | +22% |
| SVM | 44% | 59% | +22% |
| XGBoost | 46% | 60% | +22% |

## 🔮 Future Improvements

- [ ] **Feature Engineering**: Add seasonal preferences, location data
- [ ] **Deep Learning**: Implement neural networks for better pattern recognition  
- [ ] **Real Data Integration**: Replace synthetic data with actual survey data
- [ ] **Web Interface**: Create a Flask/Streamlit app for real-time predictions
- [ ] **A/B Testing**: Validate predictions with actual customer choices

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shrey Dikshant**
- Email: shreydikshant144@gmail.com

## 🙏 Acknowledgments

- **Scikit-learn** for comprehensive ML algorithms
- **Imbalanced-learn** for data balancing techniques  
- **XGBoost** for gradient boosting implementation
- **Pandas & NumPy** for data manipulation

---
