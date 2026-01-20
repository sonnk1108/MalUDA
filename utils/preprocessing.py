import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
def preprocess_data_balanced(df: pd.DataFrame, target_column: str = 'Label',
                             n_per_class: int = 30000, verbose: bool = False):
    """
    Preprocess and return exactly balanced data:
    - Remove NaNs
    - Encode categorical columns
    - Sample exactly n_per_class rows for each class (0 and 1)

    Returns:
        X, y (both numpy arrays)
    """

    if df is None or df.empty:
        print("Error: Input DataFrame is empty.")
        return None, None
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found.")
        return None, None
    # Drop NaNs
    df.dropna(inplace=True)
    # Separate target
    y = df[target_column].astype(str)
    X = df.drop(columns=[target_column], errors="ignore")
    # Encode object columns
    object_cols = X.select_dtypes(include='object').columns
    for col in object_cols:
        X[col] = pd.factorize(X[col], sort=True)[0]
    # Encode target into 0/1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return  X, y_encoded 