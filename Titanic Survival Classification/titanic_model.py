"""
titanic_model.py
────────────────
Titanic Survival Classification – Model Training & Evaluation.

Trains three classifiers (Logistic Regression, Random Forest, Gradient Boosting),
compares their performance, and provides:
  - Per-model metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Confusion matrices
  - Feature importance analysis
  - Single-passenger prediction API

All data loading is delegated to ``titanic_data_loader.py``.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from titanic_data_loader import load_titanic_data, engineer_features


# ─── Model Definitions ──────────────────────────────────────────────────────

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=6, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    ),
}


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_evaluate(csv_path: str | None = None):
    """
    End-to-end training pipeline.

    1. Loads & preprocesses data via ``titanic_data_loader.load_titanic_data``.
    2. Trains every model in ``CANDIDATE_MODELS``.
    3. Evaluates each model on the hold-out test set.
    4. Selects the best model by accuracy.

    Parameters
    ----------
    csv_path : str or None
        Optional path to the Titanic CSV.  ``None`` triggers auto-download.

    Returns
    -------
    results : dict[str, dict]
        Per-model results containing metrics, predictions, and the model object.
    best_model_name : str
        Name of the best performing model.
    meta : dict
        Auxiliary objects (preprocessor, feature_names, test labels, etc.)
    """
    # ── Load data ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_titanic_data(
        csv_path
    )

    results = {}
    best_acc = 0.0
    best_model_name = None

    # ── Train each candidate ─────────────────────────────────────────────
    for name, model in CANDIDATE_MODELS.items():
        print(f"\n{'-' * 50}")
        print(f"Training: {name}")
        print(f"{'-' * 50}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
        }

        # Pretty-print
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}\n")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    meta = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "y_test": y_test,
        "X_test": X_test,
    }

    return results, best_model_name, meta


# ─── Feature Importance ──────────────────────────────────────────────────────

def get_feature_importance(results: dict, model_name: str, feature_names: list):
    """
    Returns a sorted DataFrame of feature importances (or coefficients for LR).
    """
    model = results[model_name]["model"]

    if model_name == "Logistic Regression":
        importances = model.coef_[0]
        label = "Coefficient"
    else:
        importances = model.feature_importances_
        label = "Importance"

    df_imp = pd.DataFrame({"Feature": feature_names, label: importances})
    df_imp["AbsValue"] = df_imp[label].abs()
    df_imp.sort_values("AbsValue", ascending=False, inplace=True)
    df_imp.drop(columns="AbsValue", inplace=True)
    df_imp.reset_index(drop=True, inplace=True)
    return df_imp


# ─── Single Passenger Prediction ─────────────────────────────────────────────

def predict_passenger(passenger_dict: dict, model, preprocessor):
    """
    Predicts survival for a single passenger dictionary.

    Parameters
    ----------
    passenger_dict : dict
        Raw passenger features (Pclass, Sex, Age, SibSp, Parch, Fare,
        Embarked, Name, Cabin).
    model : fitted sklearn model
    preprocessor : fitted ColumnTransformer

    Returns
    -------
    dict with 'prediction' (0/1) and 'probability' (float).
    """
    df = pd.DataFrame([passenger_dict])
    df = engineer_features(df)
    df["Pclass"] = df["Pclass"].astype(str)

    required = ["Pclass", "Sex", "Age", "Fare", "Embarked",
                 "FamilySize", "IsAlone", "HasCabin", "Title"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required]
    processed = preprocessor.transform(df)

    pred = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    return {"prediction": int(pred), "probability": round(float(proba), 4)}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Train & evaluate all models
    results, best_name, meta = train_and_evaluate()

    print(f"\n{'=' * 60}")
    print(f"  BEST MODEL: {best_name}  (Accuracy: {results[best_name]['accuracy']:.4f})")
    print(f"{'=' * 60}")

    # 2. Classification report for the best model
    best_model = results[best_name]["model"]
    from titanic_data_loader import load_titanic_data as _reload
    X_tr, X_te, y_tr, y_te, _, _ = _reload()
    y_pred_best = best_model.predict(X_te)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_te, y_pred_best, target_names=["Not Survived", "Survived"]))

    # 3. Feature importance for best model
    imp_df = get_feature_importance(results, best_name, meta["feature_names"])
    print(f"\nFeature Importance ({best_name}):")
    print(imp_df.to_string(index=False))

    # 4. Test single-passenger prediction
    test_passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "S",
        "Name": "Chaffee, Mrs. Herbert Shingler (Carrie Toogood)",
        "Cabin": "E33",
    }

    pred = predict_passenger(
        test_passenger,
        results[best_name]["model"],
        meta["preprocessor"],
    )
    survived = "Survived" if pred["prediction"] == 1 else "Did not survive"
    print(f"\nSample Prediction: {pred}")
    print(f"  {survived} (probability: {pred['probability']:.2%})")

