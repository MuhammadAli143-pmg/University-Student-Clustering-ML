# University Student Clustering and Classification

A machine learning project that explores patterns in a synthetic university student dataset through unsupervised clustering and supervised classification. The notebook combines student, faculty, and administrative records using SQLite, preprocesses selected academic features, compares K-Means and DBSCAN clustering, and trains classifiers to reproduce K-Means-generated cluster labels.

> **Important:** The dataset is synthetic. The classification model predicts cluster labels created by K-Means; it does **not** predict student success, academic performance, graduation, or any other real-world outcome. The reported scores must not be interpreted as evidence of real-world academic prediction capability.

## Project Overview

The project demonstrates an end-to-end machine learning workflow, from loading multiple spreadsheet datasets and joining them in a relational database to clustering, classification, evaluation, model persistence, input validation, drift checks, and a Streamlit application interface.

The primary numerical features used for clustering and classification are:

- `age`
- `semester`
- `cgpa`
- `marks`
- `attendance_pct`

K-Means with three clusters supplies the target labels for the classification task. DBSCAN is evaluated separately as a density-based clustering alternative.

## Project Objectives

- Load and inspect three university-related datasets.
- Store the source datasets in SQLite and combine them with SQL `JOIN` operations.
- Clean and standardize numerical features for machine learning.
- Discover groups in the data using K-Means and DBSCAN.
- Visualize clusters using a two-component PCA projection.
- Compare Logistic Regression, Decision Tree, and Random Forest classifiers.
- Tune the selected classifier with `GridSearchCV`.
- Evaluate clustering and classification with appropriate metrics.
- Save the trained classification pipeline using Joblib.
- Validate prediction inputs against a JSON schema.
- Demonstrate prediction consistency and a simple input-shift/drift check.
- Provide a Streamlit interface for interacting with the saved model.

## Complete Machine Learning Workflow

1. **Load data:** Read `students.xlsx`, `faculty.xlsx`, and `admin_records.xlsx` with pandas.
2. **Database storage:** Write the three dataframes to `university_management.db` as SQLite tables.
3. **Relational integration:** Execute SQL `INNER JOIN` operations to create a combined dataset.
4. **Feature selection:** Select age, semester, CGPA, marks, and attendance percentage.
5. **Preprocessing:** Impute missing numerical values with the median and standardize the features.
6. **Exploratory clustering:** Use the elbow method to inspect K-Means inertia for values of *k* from 1 through 10; use `k=3` for the final K-Means clustering.
7. **Density-based clustering:** Try DBSCAN parameter combinations and select a valid trial using silhouette score and noise-point count.
8. **Cluster visualization:** Project the scaled data into two dimensions with PCA and plot K-Means and DBSCAN assignments.
9. **Cluster profiling:** Review cluster sizes and average feature values.
10. **Classification:** Use K-Means cluster assignments as labels and compare three supervised classifiers.
11. **Hyperparameter tuning:** Apply five-fold `GridSearchCV` to the best classifier from the initial comparison.
12. **Evaluation:** Calculate weighted accuracy, precision, recall, and F1-score; inspect the classification report and confusion matrix.
13. **Persistence:** Save and reload the fitted preprocessing-and-classification pipeline with Joblib, then compare predictions.
14. **Input validation:** Check required columns, numeric types, missing values, and allowed value ranges using `schema.json`.
15. **Drift demonstration:** Compare predicted cluster distributions for the test data and a deliberately perturbed version of the data.
16. **Deployment:** Use the saved model and schema in the Streamlit application.

## Dataset Descriptions

The notebook expects these three Excel workbooks in the project directory:

| File | Purpose and fields used by the notebook |
|---|---|
| `students.xlsx` | Student-level records. The SQL query selects `student_id`, `student_name`, `department`, `age`, `semester`, and `cgpa`. |
| `faculty.xlsx` | The query uses this table for the fields `course_code`, `course_name`, `marks`, `attendance_pct`, `fee_amount`, `fee_status`, and `faculty_id`. |
| `admin_records.xlsx` | The query uses this table for `faculty_id`, `faculty_name`, `designation`, `experience_years`, and `salary`. |

The notebook also exports CSV copies named `students.csv`, `faculty.csv`, and `admin_records.csv` as part of the repository's expected data assets. The source notebook reads the Excel workbooks directly; it does not show a CSV-import fallback.

The exact source dataset schemas and row counts depend on the supplied workbooks. The project uses the five numerical features listed above for clustering and classification.

## SQLite Database and SQL JOIN Operations

The notebook creates `university_management.db` with Python's built-in `sqlite3` module. It writes the three dataframes to tables named:

- `students`
- `faculty`
- `admin_records`

The combined dataset is generated with two `INNER JOIN` operations, linking student records to the second table using `student_id`, and linking that table to administrative records using `faculty_id`. In the notebook's SQL, the `faculty` table is aliased as `a` and `admin_records` as `f`.

```sql
SELECT
    s.student_id, s.student_name, s.department, s.age, s.semester, s.cgpa,
    a.course_code, a.course_name, a.marks, a.attendance_pct, a.fee_amount, a.fee_status,
    f.faculty_id, f.faculty_name, f.designation, f.experience_years, f.salary
FROM students AS s
INNER JOIN faculty AS a ON s.student_id = a.student_id
INNER JOIN admin_records AS f ON a.faculty_id = f.faculty_id
ORDER BY s.student_id;
```

An `INNER JOIN` retains records that have matching keys in the joined tables. Consequently, records without matching join keys are not included in the resulting joined dataset. The combined data is stored in the SQLite database as `clustering_results` and exported to `joined_clustering_results.xlsx`.

## Data Preprocessing

### Features

The notebook selects five numeric columns: `age`, `semester`, `cgpa`, `marks`, and `attendance_pct`.

### Missing values and scaling

- **Imputation:** `SimpleImputer(strategy="median")` fills missing feature values with each column's median for the clustering workflow.
- **Standardization:** `StandardScaler` scales the imputed features so variables with different units and ranges can contribute on a comparable scale.
- **Classification pipeline:** A scikit-learn `Pipeline` and `ColumnTransformer` apply median imputation and standardization before fitting the classifier. This keeps preprocessing attached to the saved model and ensures the same transformations are applied during prediction.

The classification pipeline is fitted on the training split. The notebook uses an 80/20 train-test split, `random_state=42`, and stratification by the K-Means-generated labels.

## Clustering

### K-Means

K-Means is used to partition the scaled feature space into three groups. The notebook plots inertia for `k=1` to `k=10` to inspect the elbow curve, then sets `optimal_k = 3`. The final cluster assignments are saved in `KMeans_Cluster`.

**Silhouette score: `0.2730`**

The silhouette score summarizes how well-separated the assigned clusters are, with higher values generally indicating more distinct clusters. A score of 0.2730 indicates some cluster structure, but also substantial overlap or within-cluster variation.

### DBSCAN

DBSCAN is a density-based clustering algorithm that can identify dense groups and mark observations that do not belong to a dense region as noise (`-1`). The notebook tests combinations of:

- `eps`: 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, and 2.0
- `min_samples`: 5, 10, and 15

Valid trials are compared using silhouette score on non-noise observations, with the number of noise points also considered. The final score is calculated excluding noise points.

**Silhouette score (excluding noise): `0.3891`**

Because noise points are excluded from this score, it should not be compared with the K-Means score as if both were calculated on exactly the same observations.

### PCA visualization and cluster profiles

PCA reduces the scaled feature matrix to two components for plotting. The notebook produces separate scatter plots for K-Means and DBSCAN and reports the explained variance ratios. It also calculates cluster sizes and mean values of the five selected features to support interpretation.

## Classification

The classification task uses `KMeans_Cluster` as its target (`y`). It therefore learns to reproduce the K-Means assignments from the five input features.

The notebook compares:

| Classifier | Configuration |
|---|---|
| Logistic Regression | `max_iter=2000`, `random_state=42` |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=150`, `random_state=42` |

Each classifier is wrapped in a preprocessing-and-classification pipeline. The initial comparison is sorted by weighted F1-score, and the top model is selected for hyperparameter tuning.

**Selected classification algorithm: Logistic Regression**

This selection is a model-selection result for the synthetic dataset and the task of reproducing K-Means labels. It is not a recommendation for predicting student outcomes.

## Hyperparameter Tuning with GridSearchCV

The selected classifier is tuned using scikit-learn's `GridSearchCV` with:

- Five-fold cross-validation (`cv=5`)
- Weighted F1-score (`scoring="f1_weighted"`)
- Parallel execution (`n_jobs=-1`)

For Logistic Regression, the notebook searches:

- `classifier__C`: 0.1, 1, 5, and 10
- `classifier__solver`: `lbfgs`

The notebook also defines parameter grids for Decision Tree and Random Forest in case either classifier is selected by the initial comparison. The best estimator returned by `GridSearchCV` becomes the final model.

## Model Evaluation

### Clustering evaluation

The notebook uses the silhouette coefficient to evaluate cluster separation:

| Clustering algorithm | Silhouette score |
|---|---:|
| K-Means (`k=3`) | **0.2730** |
| DBSCAN (excluding noise points) | **0.3891** |

### Classification evaluation

The final classification model is evaluated on the held-out test set using accuracy, weighted precision, weighted recall, and weighted F1-score.

| Metric | Result |
|---|---:|
| Selected model | Logistic Regression |
| Accuracy | **1.0000** |
| Precision (weighted) | **1.0000** |
| Recall (weighted) | **1.0000** |
| F1-score (weighted) | **1.0000** |

The notebook also generates a classification report and confusion matrix.

**Interpretation:** The perfect classification scores indicate that the selected model reproduced the generated K-Means labels on this particular test split. Since the labels are algorithmically generated from the same feature space and the data is synthetic, these scores do not demonstrate that the model can predict real student success or generalize to real university populations.

## Model Persistence with Joblib

The final fitted scikit-learn pipeline is saved as:

```text
student_cluster_classifier.joblib
```

The notebook reloads the saved artifact with `joblib.load()` and checks that predictions from the loaded model match predictions from the original in-memory model for a sample of test rows.

Load the model in Python with:

```python
import joblib

model = joblib.load("student_cluster_classifier.joblib")
```

Only load Joblib/pickle-based model files from trusted sources.

## Input Schema Validation

The notebook writes `schema.json` to describe the accepted prediction input. It requires exactly these columns:

| Column | Type | Accepted range |
|---|---|---:|
| `age` | Numeric | 16–60 |
| `semester` | Numeric | 1–16 |
| `cgpa` | Numeric | 0–4 |
| `marks` | Numeric | 0–100 |
| `attendance_pct` | Numeric | 0–100 |

The validation function checks for missing or unexpected columns, numeric data types, null values, and values outside the specified ranges. It then returns the columns in the required order as floating-point values.

Example input:

```python
import pandas as pd

new_student = pd.DataFrame([{
    "age": 21,
    "semester": 6,
    "cgpa": 3.25,
    "marks": 82,
    "attendance_pct": 88
}])
```

A successful prediction is a cluster identifier, not a student-success category.

## Data Drift Checking

The notebook demonstrates two checks:

1. **Prediction consistency:** It runs the saved model twice on the same test data and asserts that the predictions match.
2. **Input-shift demonstration:** It increases test-set `marks` values by 15 (clipped to the 0–100 range), predicts again, and compares the baseline and perturbed prediction distributions.

This is an illustrative prediction-distribution check, not a complete production drift-monitoring system. It does not implement a formal statistical drift test, automatic alerting, continuous monitoring, or retraining policy.

## Streamlit Application Deployment

The repository structure includes `streamlit_app.py`, intended to provide a user interface for submitting the five validated features and viewing the predicted K-Means cluster. The application should load `student_cluster_classifier.joblib` and validate inputs using `schema.json`.

The notebook content supplied for this project does not include the implementation of `streamlit_app.py`; therefore, the exact interface and its controls depend on the application script in the repository.

### Run locally

From the project root, install the dependencies and start Streamlit:

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL (normally `http://localhost:8501`) in the terminal. Open that address in a browser. Ensure the model and schema files are in the paths expected by the app.

## Repository Structure

```text
University_Student_ML_Project/
├── University_Student_Clustering_Final.ipynb
├── students.xlsx
├── faculty.xlsx
├── admin_records.xlsx
├── students.csv
├── faculty.csv
├── admin_records.csv
├── university_management.db
├── student_cluster_classifier.joblib
├── schema.json
├── streamlit_app.py
├── requirements.txt
├── model_card.md
├── evaluation_metrics.json
├── joined_clustering_results.xlsx
├── student_clustering_results.csv
└── README.md
```

The Excel workbooks are the notebook's input files. The database, model, schema, evaluation JSON, joined workbook, and clustering CSV are generated or saved during the workflow. The CSV source copies, Streamlit script, and model card are included in the requested repository layout.

## Installation

### Requirements

- Python 3.9 or later is recommended.
- `pip`
- The three input Excel workbooks: `students.xlsx`, `faculty.xlsx`, and `admin_records.xlsx`

### Set up a local environment

```bash
git clone <your-repository-url>
cd University_Student_ML_Project

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Google Colab Instructions

1. Upload `University_Student_Clustering_Final.ipynb` to Google Colab, or open it from your Google Drive.
2. Upload `students.xlsx`, `faculty.xlsx`, and `admin_records.xlsx` when prompted by the notebook's file-upload cell.
3. Run the notebook cells from top to bottom.
4. Review the clustering plots, cluster profiles, classifier comparison, and evaluation output.
5. Download generated artifacts from the Colab session if you want to add them to your repository. Typical outputs include `university_management.db`, `student_cluster_classifier.joblib`, `schema.json`, `evaluation_metrics.json`, `joined_clustering_results.xlsx`, and `student_clustering_results.csv`.

The notebook uses `google.colab.files.upload()` when the three Excel files are not found in the current working directory. Outside Colab, place the workbooks beside the notebook before running it.

## Technologies and Libraries

- **Python** — project programming language
- **pandas** — tabular data loading, transformation, and export
- **NumPy** — numerical operations
- **SQLite / `sqlite3`** — relational storage and SQL joins
- **scikit-learn** — preprocessing, clustering, PCA, classification, model selection, and evaluation
- **Matplotlib** — plots and visualizations
- **Seaborn** — included in the project dependency list for statistical visualization
- **Joblib** — model serialization and loading
- **OpenPyXL** — Excel workbook support through pandas
- **Streamlit** — local interactive application

## Model Results Summary

| Task | Method | Result |
|---|---|---:|
| Clustering | K-Means, 3 clusters | Silhouette: 0.2730 |
| Clustering | DBSCAN, noise excluded from score | Silhouette: 0.3891 |
| Classification | Logistic Regression | Selected model |
| Classification | Accuracy | 1.0000 |
| Classification | Weighted precision | 1.0000 |
| Classification | Weighted recall | 1.0000 |
| Classification | Weighted F1-score | 1.0000 |

## Limitations and Responsible Use

- **Synthetic data:** The dataset is synthetic and may not represent the diversity, complexity, or relationships present in real university records.
- **Not an academic prediction model:** The classifier predicts K-Means-generated cluster labels. It does not predict student success, failure, retention, graduation, or future academic performance.
- **Perfect scores need context:** The classification target is derived from the same features used as model inputs. Perfect test metrics can therefore reflect how consistently the classifier reproduces the clustering rule, rather than meaningful predictive power about students.
- **Clustering is exploratory:** Cluster IDs are arbitrary identifiers. They do not inherently represent ability levels, risk levels, or quality categories.
- **DBSCAN score qualification:** Its reported silhouette score excludes noise observations, so it is not directly comparable to a score calculated over all observations.
- **Demonstration-only drift check:** The perturbation exercise is not a substitute for formal drift detection and production monitoring.
- **Schema bounds are project rules:** The accepted ranges are validation constraints used by this project; they should not be treated as universal definitions of valid student records.
- **Privacy and fairness:** Do not use real, identifiable student records without appropriate authorization, privacy safeguards, and institutional review. Do not use cluster assignments to make consequential decisions about students.
- **Reproducibility:** Results may depend on the dataset files, library versions, and execution environment. The notebook sets random seeds for reproducibility, but exact results should be verified when rerunning the project.

This repository is intended for educational demonstration of clustering, classification, data integration, and model deployment concepts.
