"""
titanic_data_loader.py
──────────────────────
Dedicated data loader and preprocessor for the Titanic Survival Classification task.

Responsibilities:
  1. Download the Titanic CSV from a public source (caches locally).
  2. Clean raw data (handle missing values, drop irrelevant columns).
  3. Engineer features (Title, FamilySize, IsAlone, HasCabin).
  4. Build an sklearn ColumnTransformer pipeline for numerical & categorical columns.
  5. Return processed train/test splits ready for model consumption.
"""

import os
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TITANIC_CSV = os.path.join(DATA_DIR, "titanic.csv")
TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature column definitions
NUMERICAL_COLS = ["Age", "Fare", "FamilySize"]
CATEGORICAL_COLS = ["Sex", "Embarked", "Title", "Pclass"]

# Title grouping map
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Lady": "Rare",
    "the Countess": "Rare", "Capt": "Rare", "Sir": "Rare",
    "Don": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}


# ─── Download ────────────────────────────────────────────────────────────────

def download_titanic_data() -> str:
    """
    Downloads the Titanic training CSV from a public GitHub mirror.
    Returns the local file path. Skips download if the file already exists.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TITANIC_CSV):
        print(f"[titanic_data_loader] Dataset already cached at {TITANIC_CSV}")
        return TITANIC_CSV

    print(f"[titanic_data_loader] Downloading Titanic dataset from {TITANIC_URL} ...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(TITANIC_URL, headers=headers, timeout=30)
    response.raise_for_status()

    with open(TITANIC_CSV, "wb") as f:
        f.write(response.content)

    print(f"[titanic_data_loader] Saved to {TITANIC_CSV}")
    return TITANIC_CSV


# ─── Feature Engineering ─────────────────────────────────────────────────────

def _extract_title(name: str) -> str:
    """Extracts and groups the title from a passenger name string."""
    if pd.isna(name):
        return "Mr"
    title = name.split(",")[1].split(".")[0].strip()
    return TITLE_MAP.get(title, "Rare")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from the raw Titanic DataFrame.

    New columns:
      - FamilySize  = SibSp + Parch + 1
      - IsAlone     = 1 if FamilySize == 1 else 0
      - HasCabin    = 1 if Cabin is not NaN else 0
      - Title       = grouped title extracted from Name

    Drops: PassengerId, Name, Ticket, Cabin, SibSp, Parch
    """
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Title"] = df["Name"].apply(_extract_title)

    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df


# ─── Preprocessing Pipeline ──────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Builds an sklearn ColumnTransformer that:
      - Imputes + scales numerical columns.
      - Imputes + one-hot-encodes categorical columns.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS),
        ]
    )
    return preprocessor


# ─── Main Loader Function ────────────────────────────────────────────────────

def load_titanic_data(csv_path: str | None = None):
    """
    End-to-end loader for the Titanic dataset.

    Parameters
    ----------
    csv_path : str or None
        Path to the Titanic CSV.  If None, the dataset is downloaded
        automatically via ``download_titanic_data()``.

    Returns
    -------
    X_train_processed : np.ndarray   – Preprocessed training features.
    X_test_processed  : np.ndarray   – Preprocessed test features.
    y_train           : pd.Series    – Training labels.
    y_test            : pd.Series    – Test labels.
    preprocessor      : ColumnTransformer – Fitted preprocessor (for inference).
    feature_names     : list[str]    – Names of the processed feature columns.
    """
    # 1. Load raw CSV
    if csv_path is None:
        csv_path = download_titanic_data()

    raw_df = pd.read_csv(csv_path)
    print(f"[titanic_data_loader] Loaded {len(raw_df)} rows from {csv_path}")

    # 2. Separate target
    X = raw_df.drop(columns=["Survived"])
    y = raw_df["Survived"]

    # 3. Feature engineering
    X = engineer_features(X)

    # Treat Pclass as a categorical variable
    X["Pclass"] = X["Pclass"].astype(str)

    # 4. Train / test split (stratified by target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5. Build & fit preprocessor
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 6. Collect feature names after encoding
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    feature_names = list(NUMERICAL_COLS) + cat_features

    print(
        f"[titanic_data_loader] Train: {X_train_processed.shape}, "
        f"Test: {X_test_processed.shape}, Features: {len(feature_names)}"
    )

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, pp, feat = load_titanic_data()
    print(f"\nFeature names ({len(feat)}):")
    for i, name in enumerate(feat, 1):
        print(f"  {i:2d}. {name}")
    print(f"\nLabel distribution (train):\n{y_tr.value_counts()}")
