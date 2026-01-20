import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# ---------- helpers ----------


def continuous_domain_wd_and_mi(source_df, target_df, cont_features,y,
                                mi_kwargs=None, verbose=False,r=0.75):
    if mi_kwargs is None:
        mi_kwargs = {'random_state': 42}

    results = []
    y_enc = y

    for feat in cont_features:
        try:
            # --- Wasserstein Distance ---
            s_values = source_df[feat].dropna().values
            t_values = target_df[feat].dropna().values

            if len(s_values) > 0 and len(t_values) > 0:
                wd_val = wasserstein_distance(s_values, t_values)
            else:
                wd_val = np.nan
        except Exception as e:
            wd_val = np.nan
            if verbose:
                print(f"Wasserstein error for {feat}: {e}")

        try:
            # --- Mutual Information ---
            X_feat = source_df[[feat]].values
            mi_val = mutual_info_classif(
                X_feat, y_enc, discrete_features=False, **mi_kwargs
            )
            mi_with_y = float(mi_val[0])
        except Exception as e:
            mi_with_y = np.nan
            if verbose:
                print(f"MI error for {feat}: {e}")

        results.append((feat, wd_val, mi_with_y))

    df = pd.DataFrame(results, columns=['Feature', 'Domain_WD', 'MI_with_y'])

    # ---------- Robust normalization of WD ----------
    wd_median = df['Domain_WD'].median()
    wd_iqr = df['Domain_WD'].quantile(0.75) - df['Domain_WD'].quantile(0.25)

    eps = 1e-8  # numerical stability
    df['RWD'] = (df['Domain_WD'] - wd_median) / (wd_iqr + eps)

    # ---------- Thresholds ----------
    wd_threshold = df['RWD'].quantile(r) if df['RWD'].notna().any() else 0.0
    mi_cutoff = 0.01 * df['MI_with_y'].max() if df['MI_with_y'].max() > 0 else 0.0

    df['Domain_flag'] = df['RWD'] > wd_threshold        # high shift
    df['Predictive_flag'] = df['MI_with_y'] > mi_cutoff # predictive

    df['Keep'] = (~df['Domain_flag']) & (df['Predictive_flag'])

    df = df.sort_values(
        ['Domain_flag', 'Predictive_flag', 'MI_with_y', 'Domain_WD'],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)

    return df['Feature'][df['Keep']==True].values
