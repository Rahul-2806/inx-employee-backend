import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="INX Employee Performance API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "selected_features.pkl"), "rb") as f:
    selected_features = pickle.load(f)
with open(os.path.join(MODELS_DIR, "reverse_map.pkl"), "rb") as f:
    reverse_map = pickle.load(f)

# ── Label encoding maps (from raw data) ──────────────────────────────────────
DEPT_MAP = {
    "Sales": 5,
    "Human Resources": 3,
    "Development": 1,
    "Data Science": 0,
    "Research & Development": 4,
    "Finance": 2
}

ROLE_MAP = {
    "Sales Executive": 13,
    "Manager": 8,
    "Developer": 3,
    "Sales Representative": 14,
    "Human Resources": 6,
    "Senior Developer": 15,
    "Data Scientist": 1,
    "Senior Manager R&D": 16,
    "Laboratory Technician": 7,
    "Manufacturing Director": 10,
    "Research Scientist": 12,
    "Healthcare Representative": 5,
    "Research Director": 11,
    "Manager R&D": 9,
    "Finance Manager": 4,
    "Technical Architect": 17,
    "Business Analyst": 0,
    "Technical Lead": 18,
    "Delivery Manager": 2
}

OVERTIME_MAP = {"No": 0, "Yes": 1}

PERFORMANCE_LABELS = {2: "Low", 3: "Good", 4: "Excellent"}
PERFORMANCE_COLORS = {2: "#ff6b6b", 3: "#ffd60a", 4: "#34c759"}

# ── Request schema ────────────────────────────────────────────────────────────
class EmployeeInput(BaseModel):
    EmpEnvironmentSatisfaction: int       # 1-4
    EmpLastSalaryHikePercent: int         # 11-25
    YearsSinceLastPromotion: int          # 0-15
    EmpDepartment: str                    # Sales, HR, etc.
    ExperienceYearsInCurrentRole: int     # 0-18
    EmpWorkLifeBalance: int               # 1-4
    YearsWithCurrManager: int             # 0-17
    ExperienceYearsAtThisCompany: int     # 0-40
    EmpJobRole: str                       # Sales Executive, etc.
    EmpJobLevel: int                      # 1-5
    TotalWorkExperienceInYears: int       # 0-40
    OverTime: str                         # Yes / No

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "online", "project": "INX Employee Performance Predictor", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/options")
def get_options():
    return JSONResponse({
        "departments": list(DEPT_MAP.keys()),
        "job_roles": list(ROLE_MAP.keys()),
        "overtime": ["No", "Yes"],
        "ranges": {
            "EmpEnvironmentSatisfaction": {"min": 1, "max": 4, "label": "Environment Satisfaction"},
            "EmpLastSalaryHikePercent": {"min": 11, "max": 25, "label": "Last Salary Hike %"},
            "YearsSinceLastPromotion": {"min": 0, "max": 15, "label": "Years Since Last Promotion"},
            "ExperienceYearsInCurrentRole": {"min": 0, "max": 18, "label": "Experience in Current Role (yrs)"},
            "EmpWorkLifeBalance": {"min": 1, "max": 4, "label": "Work Life Balance"},
            "YearsWithCurrManager": {"min": 0, "max": 17, "label": "Years With Current Manager"},
            "ExperienceYearsAtThisCompany": {"min": 0, "max": 40, "label": "Experience at Company (yrs)"},
            "EmpJobLevel": {"min": 1, "max": 5, "label": "Job Level"},
            "TotalWorkExperienceInYears": {"min": 0, "max": 40, "label": "Total Work Experience (yrs)"},
        }
    })

@app.post("/api/predict")
def predict(data: EmployeeInput):
    try:
        # Encode categoricals
        dept_enc = DEPT_MAP.get(data.EmpDepartment)
        role_enc = ROLE_MAP.get(data.EmpJobRole)
        ot_enc   = OVERTIME_MAP.get(data.OverTime)

        if dept_enc is None:
            raise HTTPException(status_code=400, detail=f"Invalid EmpDepartment: {data.EmpDepartment}")
        if role_enc is None:
            raise HTTPException(status_code=400, detail=f"Invalid EmpJobRole: {data.EmpJobRole}")
        if ot_enc is None:
            raise HTTPException(status_code=400, detail=f"Invalid OverTime: {data.OverTime}")

        # Build feature vector in correct order
        feature_dict = {
            "EmpEnvironmentSatisfaction": data.EmpEnvironmentSatisfaction,
            "EmpLastSalaryHikePercent": data.EmpLastSalaryHikePercent,
            "YearsSinceLastPromotion": data.YearsSinceLastPromotion,
            "EmpDepartment": dept_enc,
            "ExperienceYearsInCurrentRole": data.ExperienceYearsInCurrentRole,
            "EmpWorkLifeBalance": data.EmpWorkLifeBalance,
            "YearsWithCurrManager": data.YearsWithCurrManager,
            "ExperienceYearsAtThisCompany": data.ExperienceYearsAtThisCompany,
            "EmpJobRole": role_enc,
            "EmpJobLevel": data.EmpJobLevel,
            "TotalWorkExperienceInYears": data.TotalWorkExperienceInYears,
            "OverTime": ot_enc
        }

        df = pd.DataFrame([feature_dict])[selected_features]
        scaled = scaler.transform(df)
        pred_enc = int(model.predict(scaled)[0])
        pred_rating = int(reverse_map.get(pred_enc, pred_enc))

        # Probabilities
        proba = None
        confidence = None
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(scaled)[0]
            confidence = round(float(max(proba_arr)) * 100, 1)
            proba = {
                PERFORMANCE_LABELS.get(reverse_map.get(i, i), str(i)): round(float(p) * 100, 1)
                for i, p in enumerate(proba_arr)
            }

        return JSONResponse({
            "prediction": pred_rating,
            "label": PERFORMANCE_LABELS.get(pred_rating, str(pred_rating)),
            "color": PERFORMANCE_COLORS.get(pred_rating, "#ffffff"),
            "confidence": confidence,
            "probabilities": proba,
            "top_factors": [
                "Environment Satisfaction",
                "Last Salary Hike %",
                "Years Since Last Promotion"
            ]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))