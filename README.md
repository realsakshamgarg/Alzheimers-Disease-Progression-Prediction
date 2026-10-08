# 🧠 Alzheimer's Disease Progression Prediction

A Machine Learning project for predicting Alzheimer's disease progression severity and grouping patients into risk clusters using **Random Forest Regression** and **K-Means Clustering**.

The trained Machine Learning models are integrated into a **Streamlit web application** where clinical information can be entered to generate a progression prediction and patient report.

---

## 📌 Project Overview

Alzheimer's disease progression can vary across patients. This project demonstrates how Machine Learning can be applied to clinical features to estimate a continuous disease progression severity score.

The project combines two Machine Learning approaches:

- **Supervised Learning:** Random Forest Regression
- **Unsupervised Learning:** K-Means Clustering

The workflow covers the complete process from dataset generation and model training to model evaluation, model saving, and Streamlit application integration.

> **Important:** This is an educational and research-oriented Machine Learning project. It is not a medical diagnostic system and should not be used for clinical decision-making.

---

## 🎯 Objectives

The main objectives of this project are:

1. Predict Alzheimer's disease progression severity using Machine Learning.
2. Generate a continuous progression severity score from **0 to 100**.
3. Categorize the predicted score into disease progression stages.
4. Group patients into four risk clusters using K-Means Clustering.
5. Evaluate model performance using appropriate regression metrics.
6. Check model generalization and overfitting.
7. Integrate the trained models into an interactive Streamlit application.

---

## 📊 Dataset

The project uses a **75,000-record generated/simulated clinical dataset**.

The dataset is created within the project notebook and exported as:

```text
alzheimers_clinical_data.csv
```

### Clinical Features

The model uses the following six input features:

| Feature | Description |
|---|---|
| Age | Patient age |
| Body Mass Index (BMI) | Body mass index |
| Smoking Status | Smoking status represented as a binary value |
| Alcohol Consumption | Alcohol consumption represented as a binary value |
| Mini-Mental State Examination (MMSE) Score | Cognitive assessment score |
| Functional Skill Score | Functional ability score |

### Target Variable

```text
Progression_Severity
```

The target represents a continuous progression severity score on a scale of **0 to 100**.

### Data Note

The dataset used in this project is generated/simulated for the Machine Learning workflow. It should not be interpreted as a dataset of real identifiable patients.

---

## 🧠 Machine Learning Methods

## 1. Random Forest Regression

A **Random Forest Regressor** is used as the supervised learning model.

Its purpose is to predict a continuous Alzheimer's disease progression severity score.

### Model Configuration

```text
Algorithm: Random Forest Regression
Number of Estimators: 100
Maximum Depth: 15
Random State: 42
```

The resulting severity score is mapped to four progression stages.

### Progression Severity Scale

| Progression Stage | Score Range |
|---|---:|
| Early Phase / Baseline | 0 – 24.99 |
| Mild Cognitive Impairment | 25 – 49.99 |
| Symptomatic Alzheimer's | 50 – 74.99 |
| Severe Progression | 75 – 100 |

---

## 2. K-Means Clustering

K-Means is used as the unsupervised learning component of the project.

The clinical features are first standardized using **StandardScaler** and then passed to the K-Means model.

The project uses:

```text
Number of Clusters (K): 4
```

The four clusters are associated with the following risk groups:

- Early Phase / Baseline
- Mild Cognitive Impairment
- Symptomatic Alzheimer's
- Severe Progression

The **Elbow Method** is used in the notebook to select the number of clusters.

---

## 🔬 Machine Learning Workflow

### Progression Prediction Workflow

```text
Clinical Dataset
       ↓
Data Preparation
       ↓
Feature Selection
       ↓
Train / Test Split
       ↓
Random Forest Regression
       ↓
Predicted Severity Score
       ↓
Progression Stage
```

### Patient Clustering Workflow

```text
Clinical Features
       ↓
StandardScaler
       ↓
K-Means Clustering
       ↓
4 Patient Clusters
       ↓
Risk Group
```

---

## 📈 Model Evaluation

The Random Forest Regression model produced the following reported results:

| Metric | Result |
|---|---:|
| R² Score | **98.22%** |
| Root Mean Squared Error (RMSE) | **2.14** |
| Mean Absolute Error (MAE) | **1.70** |
| 5-Fold Cross-Validation R² | **98.21%** |

### Overfitting Check

The notebook compares the training and testing performance:

| Evaluation | Result |
|---|---:|
| Training R² | **99.47%** |
| Testing R² | **98.22%** |
| Train / Test Gap | **1.26%** |

The reported train/test gap is small, indicating good generalization on this project dataset.

---

## 🔁 5-Fold Cross-Validation

The project uses **5-Fold Cross-Validation** to evaluate the consistency of the model across different data splits.

Reported fold results:

```text
Fold 1: 98.22%
Fold 2: 98.22%
Fold 3: 98.25%
Fold 4: 98.17%
Fold 5: 98.22%
```

### Cross-Validation Summary

```text
Mean R²: 98.21%
Standard Deviation: ±0.03%
```

The small variation across folds indicates consistent performance on the generated dataset.

---

## 🌲 Random Forest Prediction

The application uses the saved Random Forest model to predict a progression severity score based on the six clinical inputs.

Example workflow:

```text
Patient Clinical Information
            ↓
      Random Forest
            ↓
     Severity Score
            ↓
    Progression Stage
```

---

## 🧩 K-Means Patient Clustering

The application also uses the saved K-Means model to assign the entered patient to a cluster.

```text
Patient Clinical Features
            ↓
       StandardScaler
            ↓
        K-Means Model
            ↓
       Patient Cluster
            ↓
        Risk Group
```

---

## 🖥️ Streamlit Web Application

The trained models are integrated into a **Streamlit** web application.

The application provides a simple interface for entering:

- Age
- Body Mass Index (BMI)
- Smoking Status
- Alcohol Consumption
- Mini-Mental State Examination (MMSE) Score
- Functional Skill Score

After analysis, the application displays:

- Predicted progression severity score
- Predicted disease progression stage
- K-Means patient risk cluster
- Patient information summary
- Model performance metrics
- Disease progression severity scale
- Downloadable patient report

---

## 📄 Patient Report

After generating a prediction, the application can generate a text-based patient report containing:

- Entered clinical information
- Predicted progression severity score
- Progression stage
- K-Means cluster
- Cluster label
- Model performance metrics
- Severity scale
- Project disclaimer

The report can be downloaded directly from the Streamlit application.

---

## 📁 Project Structure

```text
Alzheimers-Disease-Progression-Prediction/
│
├── app.py
├── Disease_Progression_Prediction.ipynb
├── alzheimers_clinical_data.csv
│
├── alzheimer_rf_model.pkl
├── kmeans_model.pkl
├── kmeans_scaler.pkl
├── severity_labels.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

#### `app.py`

Streamlit application containing the user interface and prediction workflow.

#### `Disease_Progression_Prediction.ipynb`

Jupyter Notebook containing:

- Dataset generation
- Data preparation
- Model training
- Model evaluation
- Overfitting analysis
- Cross-validation
- K-Means clustering
- Cluster analysis
- Model saving

#### `alzheimers_clinical_data.csv`

Generated/simulated clinical dataset used in the project.

#### `alzheimer_rf_model.pkl`

Saved Random Forest Regression model.

#### `kmeans_model.pkl`

Saved K-Means clustering model.

#### `kmeans_scaler.pkl`

Saved StandardScaler used for K-Means clustering.

#### `severity_labels.pkl`

Saved cluster-to-severity label mapping.

#### `requirements.txt`

Python dependencies required to run the project.

#### `.gitignore`

Files and folders excluded from version control.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest Regression
- K-Means Clustering
- StandardScaler
- 5-Fold Cross-Validation

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### Web Application

- Streamlit

### Development Environment

- Jupyter Notebook
- Visual Studio Code

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Alzheimers-Disease-Progression-Prediction.git
```

### 2. Open the Project Directory

```bash
cd Alzheimers-Disease-Progression-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔄 Complete Project Pipeline

```text
              DATASET
                 ↓
       Data Generation / Preparation
                 ↓
           Feature Selection
                 ↓
           Train / Test Split
              ↙         ↘
             ↓           ↓
   Random Forest      StandardScaler
      Regression           ↓
             ↓          K-Means
             ↓             ↓
   Severity Score     Patient Cluster
             ↓             ↓
   Progression Stage    Risk Group
              \           /
               \         /
                ↓       ↓
             Streamlit
             Application
                ↓
          Patient Report
```

---

## 📌 Project Results

The project demonstrates an end-to-end Machine Learning workflow with:

- **75,000 generated clinical records**
- **Random Forest Regression** for severity prediction
- **K-Means Clustering** for patient grouping
- **R² Score of 98.22%**
- **RMSE of 2.14**
- **MAE of 1.70**
- **5-Fold Cross-Validation R² of 98.21%**
- **1.26% reported train/test performance gap**
- **Four patient risk clusters**
- **Streamlit-based prediction application**

---

## ⚠️ Limitations

This project has several important limitations:

- The dataset is generated/simulated rather than a validated clinical dataset.
- The model results are based on this project dataset and should not be treated as clinical validation.
- The application uses a limited set of six clinical features.
- External validation on independent real-world datasets has not been performed.
- The predictions are intended for educational and research demonstration purposes.

---

## 🚀 Future Scope

Possible future improvements include:

- Evaluation using validated real-world clinical datasets.
- Addition of further relevant clinical variables.
- External validation using an independent dataset.
- Further model comparison and optimization.
- More extensive clinical validation before any real-world medical application.

---

## 🧪 Research / Educational Purpose

This project demonstrates how supervised and unsupervised Machine Learning can be combined within one workflow.

The supervised component predicts a continuous severity score, while the unsupervised component groups patients based on clinical features.

The complete workflow shows the transition from:

```text
Data
  ↓
Machine Learning
  ↓
Evaluation
  ↓
Model Persistence
  ↓
Web Application
```

---

## 🔐 Model Storage

The large Random Forest model file is stored using **Git Large File Storage (Git LFS)**.

The file is:

```text
alzheimer_rf_model.pkl
```

Git LFS is used because the trained model exceeds GitHub's regular file-size handling for standard Git uploads.

---

## 📋 How to Reproduce the Project

To reproduce the Machine Learning workflow:

1. Open `Disease_Progression_Prediction.ipynb`.
2. Run the notebook from the beginning.
3. Generate the clinical dataset.
4. Train the Random Forest Regression model.
5. Evaluate the model.
6. Perform cross-validation and overfitting analysis.
7. Run K-Means clustering.
8. Save the trained models.
9. Run the Streamlit application using `app.py`.

---

## 👨‍💻 Author

### Saksham Garg

Computer Science Student  
Artificial Intelligence & Machine Learning

GitHub: [@realsakshamgarg](https://github.com/realsakshamgarg)

---

## © Copyright

**Copyright © 2026 Saksham Garg. All rights reserved.**

This repository is provided for viewing and portfolio purposes only.

No permission is granted to copy, modify, reproduce, distribute, or reuse the source code without explicit permission from the author.

---

## ⚠️ Disclaimer

This application is a Machine Learning demonstration created for educational and research purposes.

It is **not a medical diagnostic system** and must not be used to make clinical decisions.

The predictions generated by this application should not replace evaluation by a qualified healthcare professional.
