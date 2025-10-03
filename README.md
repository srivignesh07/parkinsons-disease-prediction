# Parkinson's Disease Prediction Using Machine Learning

## Project Overview
This project aims to predict Parkinson's disease in patients using machine learning algorithms. By analyzing voice measurements and biomedical features, the model can classify whether a patient is healthy or has Parkinson's disease.

## Dataset
The dataset used in this project contains biomedical voice measurements from Parkinson's patients and healthy individuals. Key features include:

- `MDVP:Fo(Hz)` – Average vocal fundamental frequency
- `MDVP:Fhi(Hz)` – Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)` – Minimum vocal fundamental frequency
- `MDVP:Jitter(%)` – Measures of variation in frequency
- `MDVP:Shimmer` – Measures of variation in amplitude
- `status` – Target variable (0: Healthy, 1: Parkinson’s)

**Source:** [UCI Machine Learning Repository - Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

## Project Steps

1. **Data Loading & Cleaning**  
   - Read the CSV dataset using pandas.
   - Check for missing values and handle them if necessary.

2. **Exploratory Data Analysis (EDA)**  
   - Visualize data distributions using `seaborn` and `matplotlib`.
   - Example:
     ```python
     import seaborn as sns
     import matplotlib.pyplot as plt

     sns.countplot(x="status", data=data, palette="viridis")
     plt.title("Count of Healthy (0) vs Parkinson’s (1)")
     plt.show()
     ```

3. **Feature Correlation Analysis**  
   - Identify relationships between features and the target variable.
   - Use heatmaps for correlation visualization:
     ```python
     plt.figure(figsize=(12,10))
     sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
     plt.show()
     ```

4. **Model Training**  
   - Split the dataset into training and testing sets.
   - Train machine learning models like:
     - Support Vector Machine (SVM)
     - Random Forest
     - Logistic Regression

5. **Model Evaluation**  
   - Evaluate model performance using metrics like:
     - Accuracy
     - Confusion Matrix
     - Classification Report

6. **Prediction**  
   - Predict new patient data and classify as Healthy (0) or Parkinson’s (1).

## Requirements
- Python 
- Libraries:
  ```text
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  
