
# ============================================================
# CARDIOVASCULAR DISEASE PREDICTION USING PYSPARK MLlib
# Academic Mini Project
#
# Algorithms:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
#
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, sum as spark_sum
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer,
    StandardScaler
)

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    r"C:\Users\reddy\Downloads\Pysparkproject"
    r"\Cardiovascular_Disease_Dataset.csv"
)

BASE_DIR = r"C:\Users\reddy\Downloads\Pysparkproject"

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("Cardiovascular Disease Prediction") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n========== CARDIOVASCULAR DISEASE PREDICTION ==========")
print("Spark Version:", spark.version)


# ============================================================
# 2. LOAD AND EXPLORE DATASET
# ============================================================

print("\n========== 2. DATASET EXPLORATION ==========")

df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

print("\nOriginal Dataset:")
df.show(10, truncate=False)

print("\nDataset Schema:")
df.printSchema()

print("\nTotal Records:", df.count())
print("Total Columns:", len(df.columns))

print("\nColumn Names:")
print(df.columns)

print("\nStatistical Summary:")
df.describe().show()

print("\nMissing Values:")

missing_values = df.select([
    spark_sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
])

missing_values.show(truncate=False)

print("\nOriginal Target Distribution:")

df.groupBy("target") \
    .count() \
    .orderBy("target") \
    .show()


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

print("\n========== 3. DATA PREPROCESSING ==========")

# Remove patient ID

if "patientid" in df.columns:
    df = df.drop("patientid")

# Convert numerical columns

numeric_columns = [
    "age",
    "restingBP",
    "serumcholestrol",
    "maxheartrate",
    "oldpeak",
    "target"
]

for c in numeric_columns:
    df = df.withColumn(
        c,
        col(c).cast("double")
    )

# Categorical columns

categorical_columns = [
    "gender",
    "chestpain",
    "fastingbloodsugar",
    "restingrelectro",
    "exerciseangia",
    "slope",
    "noofmajorvessels"
]

# Convert categorical codes into strings
# for StringIndexer and OneHotEncoder

for c in categorical_columns:
    df = df.withColumn(
        c,
        col(c).cast("string")
    )

# Remove rows with missing target or invalid target

df = df.filter(
    col("target").isin(0.0, 1.0)
)

# Handle missing categorical values

for c in categorical_columns:
    df = df.fillna({c: "Unknown"})

# Remove duplicate records

before_count = df.count()

df = df.dropDuplicates()

after_count = df.count()

print("Records before duplicate removal:", before_count)
print("Records after duplicate removal:", after_count)
print("Duplicates removed:", before_count - after_count)

# Recalculate missing values

print("\nMissing Values After Cleaning:")

df.select([
    spark_sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

print("\nCleaned Dataset:")
df.show(5)

print("Cleaned Records:", df.count())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n========== 4. FEATURE ENGINEERING ==========")

# Create Age Group

df = df.withColumn(
    "age_group",
    when(col("age").isNull(), "Unknown")
    .when(col("age") < 40, "Young")
    .when(col("age") < 60, "Middle_Aged")
    .otherwise("Senior")
)

print("\nDataset After Creating Age Group:")

df.select(
    "age",
    "age_group",
    "target"
).show(10)

# Add age group to categorical features

categorical_columns.append("age_group")

# Numerical features

feature_numeric_columns = [
    "age",
    "restingBP",
    "serumcholestrol",
    "maxheartrate",
    "oldpeak"
]

print("\nNumerical Features:")
print(feature_numeric_columns)

print("\nCategorical Features:")
print(categorical_columns)


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

print("\n========== 5. TRAIN TEST SPLIT ==========")

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training Records:", train_df.count())
print("Testing Records:", test_df.count())


# ============================================================
# 6. FEATURE TRANSFORMATION PIPELINE
# ============================================================

print("\n========== 6. FEATURE TRANSFORMATION ==========")

# Impute missing numerical values using training medians

imputed_columns = [
    c + "_imputed"
    for c in feature_numeric_columns
]

imputer = Imputer(
    inputCols=feature_numeric_columns,
    outputCols=imputed_columns,
    strategy="median"
)

# String Indexing

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_columns
]

indexed_columns = [
    c + "_index"
    for c in categorical_columns
]

# One Hot Encoding

encoded_columns = [
    c + "_encoded"
    for c in categorical_columns
]

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns,
    handleInvalid="keep"
)

# Assemble numerical and encoded features

assembler = VectorAssembler(
    inputCols=imputed_columns + encoded_columns,
    outputCol="features",
    handleInvalid="keep"
)

# Complete preprocessing pipeline

feature_pipeline = Pipeline(
    stages=[
        imputer
    ] + indexers + [
        encoder,
        assembler
    ]
)

# Fit preprocessing ONLY on training data

feature_model = feature_pipeline.fit(train_df)

train_data = feature_model.transform(train_df)
test_data = feature_model.transform(test_df)

print("\nFeature Engineering Results:")

print("Training Records:", train_data.count())
print("Testing Records:", test_data.count())

print("\nAssembled Feature Vectors:")

train_data.select(
    "features",
    "target"
).show(10, truncate=False)


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

print("\n========== 7. LOGISTIC REGRESSION ==========")

# StandardScaler is used for Logistic Regression

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=False,
    withStd=True
)

scaler_model = scaler.fit(train_data)

lr_train = scaler_model.transform(train_data)
lr_test = scaler_model.transform(test_data)

lr = LogisticRegression(
    featuresCol="scaledFeatures",
    labelCol="target",
    maxIter=50
)

lr_model = lr.fit(lr_train)

print("Logistic Regression Training Completed!")


# ============================================================
# 8. DECISION TREE
# ============================================================

print("\n========== 8. DECISION TREE ==========")

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="target",
    maxDepth=5,
    seed=42
)

dt_model = dt.fit(train_data)

print("Decision Tree Training Completed!")


# ============================================================
# 9. RANDOM FOREST
# ============================================================

print("\n========== 9. RANDOM FOREST ==========")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="target",
    numTrees=100,
    maxDepth=5,
    seed=42
)

rf_model = rf.fit(train_data)

print("Random Forest Training Completed!")


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

print("\n========== 10. MODEL EVALUATION ==========")

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="f1"
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="target",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

results = []

def evaluate_model(model, evaluation_data, model_name):

    predictions = model.transform(evaluation_data)

    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)

    print("\nModel:", model_name)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    print("\nPrediction Results:")

    predictions.select(
        "target",
        "prediction",
        "probability"
    ).show(15, truncate=False)

    # Confusion Matrix

    actual = predictions.select(
        "target"
    ).toPandas()["target"]

    predicted = predictions.select(
        "prediction"
    ).toPandas()["prediction"]

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0.0, 1.0]
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"]
    )

    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.show()

    return predictions


# Evaluate Logistic Regression

lr_predictions = evaluate_model(
    lr_model,
    lr_test,
    "Logistic Regression"
)

# Evaluate Decision Tree

dt_predictions = evaluate_model(
    dt_model,
    test_data,
    "Decision Tree"
)

# Evaluate Random Forest

rf_predictions = evaluate_model(
    rf_model,
    test_data,
    "Random Forest"
)


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

print("\n========== 11. CORRELATION HEATMAP ==========")

pandas_df = df.toPandas()

# Only numerical columns

numeric_df = pandas_df.select_dtypes(
    include="number"
)

correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Cardiovascular Disease Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 12. MODEL COMPARISON AND BEST MODEL SELECTION
# ============================================================

print("\n========== 12. MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)

print("\nFINAL MODEL PERFORMANCE:")

print(
    results_df.round(4).to_string(index=False)
)

# Save performance results

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_performance.csv"
    ),
    index=False
)

# Performance comparison graph

results_plot = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=results_plot,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Performance Score")

plt.ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_performance_comparison.png"
    ),
    dpi=300
)

plt.show()


# Select best model using ROC-AUC

best_model = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]

print("\n========== BEST MODEL SELECTION ==========")

print("Selected Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))
print("ROC-AUC:", round(best_model["ROC-AUC"], 4))


# ============================================================
# SAVE TRAINED MODELS AND PREPROCESSING
# ============================================================

print("\n========== SAVING TRAINED MODELS ==========")

feature_model.write().overwrite().save(
    os.path.join(MODEL_DIR, "feature_pipeline")
)

scaler_model.write().overwrite().save(
    os.path.join(MODEL_DIR, "logistic_scaler")
)

lr_model.write().overwrite().save(
    os.path.join(MODEL_DIR, "logistic_regression")
)

dt_model.write().overwrite().save(
    os.path.join(MODEL_DIR, "decision_tree")
)

rf_model.write().overwrite().save(
    os.path.join(MODEL_DIR, "random_forest")
)

print("Feature Pipeline Saved!")
print("Logistic Regression Saved!")
print("Decision Tree Saved!")
print("Random Forest Saved!")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 65)
print("CARDIOVASCULAR DISEASE PREDICTION COMPLETED")
print("=" * 65)

print("Total Records:", df.count())
print("Training Records:", train_data.count())
print("Testing Records:", test_data.count())

print("\nBest Algorithm:", best_model["Model"])

print("\nAll Outputs Saved In:")
print(OUTPUT_DIR)

print("\nAll Models Saved In:")
print(MODEL_DIR)

spark.stop()

print("\nSPARK SESSION CLOSED SUCCESSFULLY!")
