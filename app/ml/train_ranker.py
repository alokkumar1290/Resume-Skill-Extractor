"""Lightweight ML model training for resume ranking.
Train a LogisticRegression classifier to predict hire probability using features:
- cgpa
- num_tech_skills
- Np. of project 

Usage: python -m app.ml.train_ranker
Produces: model.pkl in same directory
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import Session

from app.database.models import SessionLocal, Resume

MODEL_PATH = Path(__file__).with_suffix(".pkl")


def _build_features(resume: Resume) -> tuple[list[float], int]:
    """Return features list and label (hired)"""
    cgpa = resume.cgpa or 0.0
    try:
        skills = json.loads(resume.skills)['technical']
        num_skills = len(skills)
    except Exception:
        num_skills = 0
    try:
        exp = json.loads(resume.experience)
        exp_years = len(exp)
    except Exception:
        exp_years = 0
    label = resume.hired
    return [cgpa, num_skills, exp_years], label


def train():
    db: Session = SessionLocal()
    resumes = db.query(Resume).all()
    if not resumes:
        print("No data to train.")
        return
    X, y = zip(*[_build_features(r) for r in resumes])
    X = np.array(X)
    y = np.array(y)
    model = LogisticRegression()
    model.fit(X, y)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    print("Model trained and saved to", MODEL_PATH)

if __name__ == "__main__":
    train()
