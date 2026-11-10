# University Student Clustering and Classification

An educational machine-learning project that explores a synthetic university dataset using clustering and classification. It combines student, faculty, and administrative records with SQLite, applies K-Means and DBSCAN, and trains a classifier to reproduce K-Means cluster assignments.

> **Important:** This project uses synthetic data. The classifier predicts K-Means-generated cluster IDs; it does not predict student success, academic performance, graduation, or other real-world outcomes. Do not use cluster assignments to make decisions about students.

## Project overview

The workflow demonstrates data integration, preprocessing, unsupervised learning, supervised classification, evaluation, model persistence, input validation, and an intended Streamlit interface.

Features used for clustering and classification:

- `age`
- `semester`
- `cgpa`
- `marks`
- `attendance_pct`

K-Means with three clusters provides the labels for the classification task. DBSCAN is evaluated separately as an alternative clustering method.

## Workflow

1. Load `students.xlsx`, `faculty.xlsx`, and `admin_records.xlsx` with pandas.
2. Save the input dataframes to SQLite tables and join the records with SQL.
3. Select the five numeric features, impute missing values with the median, and standardize them.
4. Explore K-Means cluster counts with the elbow method and fit the final three-cluster model.
5. Test DBSCAN parameter combinations and visualize cluster assignments using PCA.
6. Compare Logistic Regression, Decision Tree, and Random Forest models trained to reproduce K-Means labels.
7. Tune the selected classifier with five-fold `GridSearchCV` and evaluate it on a held-out test split.
8. Save the fitted preprocessing and classification pipeline with Joblib, validate input data with `schema.json`, and demonstrate prediction consistency and input shift.

## Dataset files

| File | Description |
|---|---|
| `students.xlsx` | Student records, including student ID, department, age, semester, and CGPA. |
| `faculty.xlsx` | Course-related records, including marks, attendance, fees, and faculty ID. |
| `admin_records.xlsx` | Administrative and faculty records, including faculty ID, designation, experience, and salary. |

The notebook joins the records using `student_id` and `faculty_id`. Since these are inner joins, rows without matching keys are excluded. The exact schemas and row counts depend on the supplied workbooks.

## Results

The README supplied with the project reports the following results. Re-run the notebook to verify them for your data and environment.

| Task | Method | Reported result |
|---|---|---:|
| Clustering | K-Means, 3 clusters | Silhouette score: 0.2730 |
| Clustering | DBSCAN, excluding noise | Silhouette score: 0.3891 |
| Classification | Logistic Regression | Selected model |
| Classification | Held-out test metrics | Accuracy, weighted precision, recall, and F1: 1.0000 |

The DBSCAN silhouette score excludes noise points, so it is not directly comparable to the K-Means score. Perfect classification metrics reflect reproducing labels generated from the same input features; they are not evidence of real-world student-outcome prediction.

## Installation and use

Requirements: Python 3.9 or later, pip, and the three Excel input workbooks listed above.

```bash
git clone https://github.com/MuhammadAli143-pmg/University-Student-Clustering-ML.git
cd University-Student-Clustering-ML
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and run the notebook from top to bottom:

```bash
pip install -r requirements.txt
```

Place the Excel workbooks in the project directory before running locally. In Google Colab, upload the notebook and provide the workbooks when prompted.

If the model artifact and Streamlit script are present and configured, start the interface with:

```bash
streamlit run streamlit_app.py
```

## Model inputs

The project schema accepts these numeric inputs:

| Field | Allowed range |
|---|---:|
| `age` | 16–60 |
| `semester` | 1–16 |
| `cgpa` | 0–4 |
| `marks` | 0–100 |
| `attendance_pct` | 0–100 |

These are project-specific validation limits, not universal standards. A prediction is a cluster identifier only.

## Outputs

Depending on which notebook cells have been run and which files are included, outputs may include:

- `university_management.db`
- `student_cluster_classifier.joblib`
- `schema.json`
- `evaluation_metrics.json`
- `joined_clustering_results.xlsx`
- `student_clustering_results.csv`

Only load Joblib/pickle model files from trusted sources. The input-shift example is an educational demonstration, not a production drift-monitoring system.

## Technologies

Python, pandas, NumPy, SQLite, scikit-learn, Matplotlib, Seaborn, Joblib, OpenPyXL, and Streamlit.

## Responsible use and limitations

- The data is synthetic and may not represent real university populations.
- Cluster IDs are arbitrary and should not be interpreted as ability, risk, or quality categories.
- The classifier learns to reproduce clustering assignments; it is not an academic prediction model.
- Do not use real, identifiable student data without appropriate authorization, privacy safeguards, and institutional review.
- Results may vary with the input data, library versions, and execution environment.

This repository is intended for educational demonstration of clustering, classification, data integration, and model deployment concepts.
