import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_excel(
    r"C:\DATA SCIENCE PROJECTS\IABAC\INX_Employee_Performance\data\raw\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls"
)

print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print("Target distribution:\n", df['PerformanceRating'].value_counts())

# ── Encode categoricals ───────────────────────────────────────────────────────
le_dept = LabelEncoder()
le_role = LabelEncoder()
df['EmpDepartment'] = le_dept.fit_transform(df['EmpDepartment'])
df['EmpJobRole']    = le_role.fit_transform(df['EmpJobRole'])
df['OverTime']      = df['OverTime'].map({'Yes': 1, 'No': 0})

# ── Features & target ─────────────────────────────────────────────────────────
selected_features = [
    'EmpEnvironmentSatisfaction', 'EmpLastSalaryHikePercent',
    'YearsSinceLastPromotion', 'EmpDepartment', 'ExperienceYearsInCurrentRole',
    'EmpWorkLifeBalance', 'YearsWithCurrManager', 'ExperienceYearsAtThisCompany',
    'EmpJobRole', 'EmpJobLevel', 'TotalWorkExperienceInYears', 'OverTime'
]

X = df[selected_features]
y = df['PerformanceRating']

# ── Label map ─────────────────────────────────────────────────────────────────
classes = sorted(y.unique())
label_map   = {c: i for i, c in enumerate(classes)}
reverse_map = {i: c for i, c in enumerate(classes)}
y_enc = y.map(label_map)

print("Classes:", classes)
print("Label map:", label_map)
print("Reverse map:", reverse_map)

# ── Train ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)

acc = model.score(X_test_sc, y_test)
print(f"✅ Accuracy: {acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
import os
OUT = r"C:\DATA SCIENCE PROJECTS\IABAC\inx-employee-backend\models"
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "best_model.pkl"), "wb") as f:    pickle.dump(model, f)
with open(os.path.join(OUT, "scaler.pkl"), "wb") as f:        pickle.dump(scaler, f)
with open(os.path.join(OUT, "selected_features.pkl"), "wb") as f: pickle.dump(selected_features, f)
with open(os.path.join(OUT, "reverse_map.pkl"), "wb") as f:   pickle.dump(reverse_map, f)

print("✅ All models saved to", OUT)
print("Dept classes:", list(le_dept.classes_))
print("Role classes:", list(le_role.classes_))
