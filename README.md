# Cardiovascular Disease Prediction Using PySpark MLlib

## Project Overview

This project implements a machine learning-based Cardiovascular Disease Prediction system using **Apache PySpark and Spark MLlib**.

The system analyzes patient health attributes and applies three machine learning classification algorithms to predict whether a patient belongs to the No Disease or Disease class.

The project also performs data preprocessing, feature engineering, model evaluation, correlation analysis, and model comparison.

## Objectives

- Analyze cardiovascular disease data using PySpark.
- Perform data cleaning and preprocessing.
- Handle missing values and duplicate records.
- Perform feature engineering using patient age groups.
- Train multiple machine learning classification algorithms.
- Evaluate and compare model performance.
- Identify the best-performing model based on ROC-AUC.

## Dataset Description

- **Dataset:** Cardiovascular Disease Dataset
- **File:** `Cardiovascular_Disease_Dataset.csv`
- **Total Records:** 1,000
- **Total Columns:** 14
- **Target Column:** `target`

### Dataset Features

| Feature | Description |
|---|---|
| patientid | Patient identifier |
| age | Patient age |
| gender | Gender category |
| chestpain | Chest pain category |
| restingBP | Resting blood pressure |
| serumcholestrol | Serum cholesterol |
| fastingbloodsugar | Fasting blood sugar category |
| restingrelectro | Resting electrocardiogram |
| maxheartrate | Maximum heart rate |
| exerciseangia | Exercise-related angina |
| oldpeak | Oldpeak measurement |
| slope | Slope category |
| noofmajorvessels | Number of major vessels |
| target | Prediction class |

**Target values:**
- `0` – No Disease
- `1` – Disease

Original target distribution:
- No Disease: 420 records
- Disease: 580 records

## Technologies Used

- Python
- Apache Spark 3.5.9
- PySpark
- Spark MLlib
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Java 17

## Machine Learning Algorithms

Three classification algorithms are implemented:

1. **Logistic Regression** – Predicts the probability of belonging to a target class.
2. **Decision Tree Classifier** – Uses decision rules to classify patient records.
3. **Random Forest Classifier** – Combines multiple decision trees to make predictions.

## Project Methodology

1. Spark Session Initialization
2. Dataset Loading and Exploration
3. Data Preprocessing
4. Feature Engineering
5. Training and Testing Split (80:20)
6. Feature Transformation
7. Model Training
8. Model Evaluation
9. Correlation Heatmap
10. Model Comparison and Selection
11. Saving Trained Models

### Feature Engineering

The age feature is divided into three groups:

- Young: Below 40
- Middle_Aged: 40 to below 60
- Senior: 60 and above

Categorical features are transformed using:
- StringIndexer
- OneHotEncoder

Numerical features are processed using:
- Imputer (Median Strategy)

VectorAssembler combines the processed features into a single feature vector.

## Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Experimental Results

The following results were obtained from the test dataset.

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 96.91% | 96.91% | 96.91% | 96.91% | 0.9946 |
| Decision Tree | 95.06% | 95.25% | 95.06% | 95.09% | 0.9705 |
| Random Forest | 95.06% | 95.06% | 95.06% | 95.06% | 0.9949 |

**Best Model Based on ROC-AUC: Random Forest**

- Accuracy: 95.06%
- ROC-AUC: 0.9949

Logistic Regression achieved the highest accuracy of 96.91%, while Random Forest achieved the highest ROC-AUC of 0.9949.

## Project Structure

```text
Cardiovascular-Disease-Prediction/
│
├── cardiovascular_prediction.py
├── Cardiovascular_Disease_Dataset.csv
├── requirements.txt
├── README.md
│
├── models/
│   └── README.md
│
└── output/
    └── README.md
```

The `models` and `output` directories are generated or populated when the Python script executes successfully.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Cardiovascular-Disease-Prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Dataset Path

Open `cardiovascular_prediction.py` and update `DATA_PATH` to the location of your downloaded CSV file.

For example:

```python
DATA_PATH = "Cardiovascular_Disease_Dataset.csv"
```

Make sure the CSV file is in the project directory.

### 5. Run the Project

```bash
python cardiovascular_prediction.py
```

The script will load the dataset, preprocess the data, train the three classification models, evaluate their performance, generate visualizations, and save the trained models.

## Generated Outputs

The project generates the following outputs:

- Logistic Regression confusion matrix
- Decision Tree confusion matrix
- Random Forest confusion matrix
- Cardiovascular disease correlation heatmap
- Model performance comparison chart
- Model performance CSV file

## Saved Models

The trained models and preprocessing components are saved inside the `models/` directory:

- Feature preprocessing pipeline
- Logistic Regression scaler
- Logistic Regression model
- Decision Tree model
- Random Forest model

## Conclusion

This project demonstrates how Apache PySpark and Spark MLlib can be used to build a complete machine learning classification workflow for cardiovascular disease prediction.

Three classification algorithms were implemented and compared. Logistic Regression achieved the highest accuracy, while Random Forest achieved the highest ROC-AUC and was selected based on the project's model-selection criterion.

**Note:** This is an academic machine learning project. Its predictions are not medically validated and should not be used as a substitute for professional medical assessment.

## Author

Padala Phani Bhaskara Reddy

