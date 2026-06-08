"""
=============================================================
 CICIDS2017 — Random Forest IDS Pipeline + Visualizations
=============================================================

Pipeline:
  1. Load dataset
  2. Clean column names
  3. Handle missing values
  4. Handle infinite values
  5. Remove duplicates
  6. Remove very small classes
  7. Visualize attack distribution
  8. Encode labels
  9. Feature selection
 10. Stratified sampling
 11. Train/Test split
 12. Train Random Forest
 13. Evaluate model
 14. ROC Curve
 15. Feature importance
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 
import warnings

warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    recall_score 
)
from sklearn.feature_selection import VarianceThreshold


# ============================================================
# STEP 1 — LOAD DATASET
# ============================================================

print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

CSV_PATH = "cicids2017_cleaned.csv"

df = pd.read_csv(CSV_PATH)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nFirst 3 rows:")
print(df.head(3))


# ============================================================
# STEP 2 — CLEAN COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Cleaning Column Names")
print("=" * 60)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_per_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)

print("\nColumns:")
print(df.columns.tolist())

# Detect label column
LABEL_COL = None

for candidate in ["attack_type", "label", "class", "target"]:
    if candidate in df.columns:
        LABEL_COL = candidate
        break

if LABEL_COL is None:
    raise ValueError("Could not detect label column.")

print(f"\nLabel column detected: {LABEL_COL}")


# ============================================================
# STEP 3 — HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Handling Missing Values")
print("=" * 60)

missing_before = df.isnull().sum().sum()

print(f"Missing values before: {missing_before:,}")

# Drop columns with >50% missing
cols_before = df.shape[1]

df.dropna(
    thresh=int(0.5 * len(df)),
    axis=1,
    inplace=True
)

print(f"Columns dropped: {cols_before - df.shape[1]}")

# Numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if LABEL_COL in num_cols:
    num_cols.remove(LABEL_COL)

# Fill NaNs with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print(f"Missing values after: {df.isnull().sum().sum():,}")


# ============================================================
# STEP 4 — HANDLE INFINITE VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Handling Infinite Values")
print("=" * 60)

inf_count = np.isinf(df[num_cols]).sum().sum()

print(f"Infinite values found: {inf_count:,}")

df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print(f"Infinite values after fix: {np.isinf(df[num_cols]).sum().sum()}")


# ============================================================
# STEP 5 — REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Removing Duplicates")
print("=" * 60)

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"Duplicates removed: {before - after:,}")
print(f"Remaining rows    : {after:,}")


# ============================================================
# STEP 6 — REMOVE VERY SMALL CLASSES
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Removing Very Small Classes")
print("=" * 60)

classes_to_remove = [
    "Bots",
    "Web Attacks"
]

before_filter = len(df)

df = df[~df["attack_type"].isin(classes_to_remove)]

after_filter = len(df)

print(f"Rows removed: {before_filter - after_filter:,}")

print("\nRemaining classes:")
print(df["attack_type"].value_counts())


# ============================================================
# STEP 7 — ATTACK DISTRIBUTION VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Attack Distribution Visualization")
print("=" * 60)

# Count each class
attack_distribution = df["attack_type"].value_counts()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

attack_distribution.plot(
    kind="bar",
    ax=ax
)

# Titles and labels
ax.set_title(
    "Distribution of Traffic Types in CICIDS2017 Dataset",
    fontsize=14
)

ax.set_xlabel("Traffic Type")
ax.set_ylabel("Number of Records")

# Rotate labels
plt.xticks(rotation=20)

# Add values on top of bars
for i, v in enumerate(attack_distribution.values):
    ax.text(
        i,
        v + 1000,
        f"{v:,}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig("attack_distribution.png", dpi=150)

plt.close()

print("Attack distribution chart saved -> attack_distribution.png")

# ============================================================
# STEP 8 — ENCODE LABELS
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Encoding Labels")
print("=" * 60)

le = LabelEncoder()

df[LABEL_COL] = le.fit_transform(
    df[LABEL_COL].astype(str).str.strip()
)

print("\nEncoded classes:")

for cls, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"{code} -> {cls}")


# ============================================================
# STEP 9 — FEATURE SELECTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Feature Selection")
print("=" * 60)

X = df.drop(columns=[LABEL_COL])

y = df[LABEL_COL]

# Remove non numeric columns
non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()

if non_num:
    print(f"\nDropping non numeric columns: {non_num}")
    X.drop(columns=non_num, inplace=True)

# Remove zero variance features
vt = VarianceThreshold(threshold=0.0)

vt.fit(X)

zero_var_cols = X.columns[~vt.get_support()].tolist()

print(f"\nZero variance features removed: {len(zero_var_cols)}")

X = X[X.columns[vt.get_support()]]

# Remove highly correlated features
corr_matrix = X.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_cols = [
    col for col in upper.columns
    if any(upper[col] > 0.98)
]

print(f"Highly correlated features removed: {len(high_corr_cols)}")

X.drop(columns=high_corr_cols, inplace=True)

print(f"Remaining features: {X.shape[1]}")


# ============================================================
# STEP 10 — BALANCED SAMPLING
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Balanced Sampling")
print("=" * 60)

SAMPLES_PER_CLASS = 15_000  

df_fs = X.copy()
df_fs[LABEL_COL] = y

sampled_parts = []

for label_value, group in df_fs.groupby(LABEL_COL):
    n_samples = min(SAMPLES_PER_CLASS, len(group))
    sampled_group = group.sample(n=n_samples, random_state=42)
    sampled_parts.append(sampled_group)

df_sampled = pd.concat(sampled_parts)
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_sampled.drop(columns=[LABEL_COL])
y = df_sampled[LABEL_COL]

print(f"\nSampled rows: {len(X):,}")

print("\nSampled class distribution (BALANCED):")

# Create mapping from encoded number to class name
label_mapping = dict(zip(le.transform(le.classes_), le.classes_))

# Use the mapping to decode labels
decoded_labels = y.map(label_mapping)

print(decoded_labels.value_counts())

# ============================================================
# STEP 11 — TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# STEP 12 — TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: Training Random Forest")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=500,        # Much more trees (was 200)
    max_depth=None,          # No limit — full depth
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample"
)

print("\nTraining model...")

rf.fit(X_train, y_train)

print("Training complete!")

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ============================================================
# TRAIN DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("Training Decision Tree")
print("=" * 60)

dt = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=30,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced"
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

print(f"Decision Tree Accuracy: {dt_acc*100:.2f}%")

# ============================================================
# TRAIN KNN
# ============================================================

print("\n" + "=" * 60)
print("Training KNN")
print("=" * 60)

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)

print(f"KNN Accuracy: {knn_acc*100:.2f}%")

# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Random Forest : {rf_acc*100:.2f}%")
print(f"Decision Tree : {dt_acc*100:.2f}%")
print(f"KNN           : {knn_acc*100:.2f}%")

# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"\nMean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Std Deviation    : {cv_scores.std() * 100:.4f}%")

# ============================================================
# STEP 13 — MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: Model Evaluation")
print("=" * 60)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[str(c) for c in le.classes_]
    )
)

# Confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=True,
    cmap="Blues"
)

ax.set_title(
    "Confusion Matrix — Random Forest on CICIDS2017",
    fontsize=14
)

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)

plt.close()

print("\nConfusion matrix saved -> confusion_matrix.png")

# ============================================================
# DETECTION RATE (RECALL) FOR ALL CLASSES
# ============================================================

print("\n" + "=" * 60)
print("DETECTION RATE (RECALL) FOR ALL CLASSES")
print("=" * 60)

# Compute recall per class
recalls = recall_score(y_test, y_pred, average=None)

# Build clean mapping: encoded -> original label
class_mapping = dict(zip(range(len(le.classes_)), le.classes_))

for class_id, recall_value in enumerate(recalls):

    class_name = class_mapping[class_id]
    detection_rate = recall_value * 100

    print(f"{class_name:<20} : {detection_rate:.2f}%")

# ============================================================
# FPR VISUALIZATION
# ============================================================

fpr_values = []

n_classes = cm.shape[0]

for i in range(n_classes):

    FP = cm[:, i].sum() - cm[i, i]
    TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])

    FPR = FP / (FP + TN + 1e-10)

    fpr_values.append(FPR * 100)

plt.figure(figsize=(10, 5))

plt.bar(le.classes_, fpr_values)

plt.title("False Positive Rate per Class")
plt.xlabel("Class")
plt.ylabel("FPR (%)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig("false_positive_rate.png", dpi=150)

plt.close()

print("\nFalse Positive Rate per Class:")

for cls, value in zip(le.classes_, fpr_values):
    print(f"{cls:<20} : {value:.4f}%")

# ============================================================
# STEP 14 — ROC CURVE
# ============================================================

print("\n" + "=" * 60)
print("STEP 14: ROC Curve")
print("=" * 60)

# Binarize labels
y_test_bin = label_binarize(
    y_test,
    classes=np.unique(y)
)

# Predict probabilities
y_score = rf.predict_proba(X_test)

# Number of classes
n_classes = y_test_bin.shape[1]

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

for i in range(n_classes):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{le.classes_[i]} (AUC = {roc_auc:.3f})"
    )

# Random guess line
ax.plot([0, 1], [0, 1], linestyle="--")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve — Random Forest")

ax.legend(loc="lower right")

plt.tight_layout()

plt.savefig("roc_curve.png", dpi=150)

plt.close()

print("ROC curve saved -> roc_curve.png")


# ============================================================
# STEP 15 — FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: Feature Importance")
print("=" * 60)

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

top20 = importances.sort_values(
    ascending=False
).head(20)

fig, ax = plt.subplots(figsize=(10, 7))

top20.sort_values().plot(
    kind="barh",
    ax=ax,
    color="steelblue"
)

ax.set_title(
    "Top 20 Feature Importances",
    fontsize=13
)

ax.set_xlabel("Importance Score")

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=150)

plt.close()

print("\nTop 20 Features:")

for feat, imp in top20.items():
    print(f"{feat:<40} {imp:.4f}")

print("\nFeature importance plot saved -> feature_importance.png")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)

print("\nGenerated files:")


import joblib

joblib.dump(rf, "rf_ids_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved successfully!")


print(" - attack_distribution.png")
print(" - confusion_matrix.png")
print(" - false_positive_rate.png")
print(" - roc_curve.png")
print(" - feature_importance.png")