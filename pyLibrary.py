def univariate(df):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  df_results = pd.DataFrame(columns=["Data Type", "Count", "Missing", "Unique", "Mode", "Min", "Q1", "Median",
                                     "Q3", "Max", "Mean", "Std", "Skew", "Kurt"])

  for col in df.columns:
    df_results.loc[col, "Data Type"] = df[col].dtype
    df_results.loc[col, "Count"] = df[col].count()
    df_results.loc[col, "Missing"] = df[col].isna().sum()
    df_results.loc[col, "Unique"] = df[col].nunique()
    df_results.loc[col, "Mode"] = df[col].mode()[0]

    if df[col].dtype in ["int64", "float64"]:
      df_results.loc[col, "Min"] = df[col].min()
      df_results.loc[col, "Q1"] = df[col].quantile(0.25)
      df_results.loc[col, "Median"] = df[col].median()
      df_results.loc[col, "Q3"] = df[col].quantile(0.75)
      df_results.loc[col, "Max"] = df[col].max()
      df_results.loc[col, "Mean"] = df[col].mean()
      df_results.loc[col, "Std"] = df[col].std()
      df_results.loc[col, "Skew"] = df[col].skew()
      df_results.loc[col, "Kurt"] = df[col].kurt()

      # Check if column is NOT boolean 0/1
      unique_vals = set(df[col].dropna().unique())
      is_boolean = unique_vals.issubset({0, 1})
      
      if not is_boolean:
        # Create stacked plot: horizontal box on top, histogram with KDE underneath (shared x-axis)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(10, 6),
                                            gridspec_kw={"height_ratios": (.15, .85)})
        sns.set_style('ticks')

        flierprops = dict(marker='o', markersize=4, markerfacecolor='none', linestyle='none', markeredgecolor='gray')
        sns.boxplot(data=df, x=col, ax=ax_box, fliersize=4, saturation=0.50, width=0.50, linewidth=0.5, flierprops=flierprops)
        sns.histplot(data=df, x=col, ax=ax_hist, kde=True, color="orange")

        ax_box.set(yticks=[])
        ax_box.set(xticks=[])
        ax_box.set_xlabel('')
        ax_box.set_ylabel('')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_xlabel(col)
        plt.suptitle(f'Box Plot and Distribution for {col}', y=1.02)
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True, bottom=True)
        plt.tight_layout()
        plt.show()
    else:
      # Prepare for categorical plots
      plt.figure(figsize=(10, 6))
      ax = sns.countplot(data=df, x=col)
      plt.title(f'Count Plot for {col}')
      plt.xlabel(col)
      plt.ylabel('Count')
      plt.xticks(rotation=45, ha='right')
      
      # Add percentage labels above each bar
      total = len(df[col].dropna())
      for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.text(p.get_x() + p.get_width() / 2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom')
      
      plt.tight_layout()
      plt.show()

  return df_results

def drop_columns(df):
  import pandas as pd

  for col in df.columns:
    if df[col].nunique() == 1:
      df.drop(col, axis=1, inplace=True)

    # Drop any column where the number of unique values equals the number of rows
    # and if the column is not numeric
    elif df[col].nunique() == df.shape[0]:
      if df[col].dtype == 'object':
        df.drop(col, axis=1, inplace=True)

  return df

def wrangle_basic(df, equivalence_mapping=None):
  """
  Clean categorical text fields to eliminate data quality issues from inconsistent data entry.
  Creates new columns with '_clean' suffix; original columns are unchanged. Row count is preserved.
  Semantically identical values (e.g. 'UT', 'ut', 'Utah', 'utah') are mapped to one canonical string.

  Values are grouped by normalized form (strip + lower); within each group the most frequent
  original value is used as the canonical form. Optionally pass equivalence_mapping to merge
  variants that differ beyond case/whitespace (e.g. state abbrev -> full name):
  equivalence_mapping={ "state": {"UT": "Utah", "ut": "Utah", "Utah": "Utah", "utah": "Utah"} }
  """
  import pandas as pd
  from collections import defaultdict

  out = df.copy()
  for col in out.columns:
    if out[col].dtype != "object" and not pd.api.types.is_string_dtype(out[col]):
      continue
    s = out[col].astype(object)
    # Optional: apply user-provided equivalence mapping first (e.g. UT/ut/Utah/utah -> Utah)
    if equivalence_mapping is not None and col in equivalence_mapping:
      em = equivalence_mapping[col]
      s = s.map(lambda x: em.get(x, x) if pd.notna(x) else x)
    # Group values by normalized key (strip + lower) to find case/whitespace variants
    key_to_raws = defaultdict(list)
    for raw in s.dropna().unique():
      key = str(raw).strip().lower()
      key_to_raws[key].append(raw)
    # For each group, choose canonical form = most frequent value in that group
    counts = s.value_counts()
    raw_to_canonical = {}
    for key, raws in key_to_raws.items():
      canonical = max(raws, key=lambda r: counts.get(r, 0))
      for r in raws:
        raw_to_canonical[r] = canonical
    cleaned = s.map(lambda x: raw_to_canonical.get(x, x) if pd.notna(x) else x)
    out[col + "_clean"] = cleaned
  return out

def add_datetime_features(df):
  """
  Parse messy datetime strings from text columns and add time-based analytical features.
  Parses at least stop_datetime_raw and scheduled_window_start_raw (handles inconsistent
  formats, 12/24h, timezone tokens). Adds: day_of_week (0=Monday), weekend (0/1),
  lateness_min (minutes late vs scheduled window start; >= 0, comparable to actual_arrival_min).
  """
  import pandas as pd
  import numpy as np
  from dateutil import parser as dateutil_parser

  out = df.copy()
  datetime_cols = ["stop_datetime_raw", "scheduled_window_start_raw"]
  parsed_names = ["stop_datetime_parsed", "scheduled_window_start_parsed"]

  def safe_parse(ser):
    def parse_one(x):
      if pd.isna(x) or x is None or str(x).strip() == "":
        return pd.NaT
      try:
        dt = dateutil_parser.parse(str(x).strip())
        if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
          dt = dt.replace(tzinfo=None)
        return dt
      except Exception:
        return pd.NaT
    return ser.map(parse_one)

  for col, new_name in zip(datetime_cols, parsed_names):
    if col in out.columns:
      out[new_name] = pd.to_datetime(safe_parse(out[col]), errors="coerce")

  # Time-based features from scheduled window start (reference for "delivery day")
  ref = "scheduled_window_start_parsed"
  if ref in out.columns:
    out["day_of_week"] = out[ref].dt.weekday  # Monday=0, Sunday=6
    out["weekend"] = (out["day_of_week"] >= 5).astype(int)
    # Lateness: minutes after scheduled window start (same scale as actual_arrival_min), never negative
    scheduled_start_min = out[ref].dt.hour * 60 + out[ref].dt.minute
    if "actual_arrival_min" in out.columns:
      out["lateness_min"] = (out["actual_arrival_min"].astype("float64") - scheduled_start_min).clip(lower=0)
    else:
      out["lateness_min"] = np.nan
  return out

def bin_categories(df, columns=None, min_percent=0.05, min_count=15, drop_below_threshold_other=False):
  import pandas as pd

  # If columns is None or empty list, apply to every column; otherwise only the listed columns
  cols_to_process = list(df.columns) if (columns is None or len(columns) == 0) else columns

  for col in cols_to_process:
    n_total = len(df)
    if col not in df.columns:
      continue  # skip if column name not in dataframe
    if df[col].dtype == 'object':
      value_counts = df[col].value_counts()
      # Keep a category if it meets EITHER threshold (count >= min_count OR percent >= min_percent)
      to_bin = []
      for val, count in value_counts.items():
        pct = count / n_total
        if count < min_count and pct < min_percent:
          to_bin.append(val)
      df[col] = df[col].replace(to_bin, 'Other')

      # Optionally drop rows where 'Other' doesn't meet either threshold
      if drop_below_threshold_other and 'Other' in df[col].values:
        other_count = (df[col] == 'Other').sum()
        other_pct = other_count / len(df)
        if other_count < min_count and other_pct < min_percent:
          df.drop(df[df[col] == 'Other'].index, inplace=True)

  return df

def bin_rare_categories(df, cols=None, min_prop=0.05, suffix="_binned"):
  """
  Consolidate infrequent categories into 'Other' to reduce cardinality.
  Creates new columns with the given suffix; original columns are unchanged. All rows preserved.

  cols: None (all categorical columns), a single column name (str), or list of column names.
  min_prop: categories with frequency (proportion of rows) below this are binned into 'Other' (default 0.05).
  suffix: appended to column names for the binned versions (default '_binned').
  """
  import pandas as pd

  out = df.copy()
  n_total = len(out)
  if n_total == 0:
    return out

  # Resolve column list: None -> all categorical; str -> [str]; list -> as-is
  if cols is None:
    cols_to_process = [
      c for c in out.columns
      if out[c].dtype == "object" or pd.api.types.is_string_dtype(out[c])
    ]
  elif isinstance(cols, str):
    cols_to_process = [cols] if cols in out.columns else []
  else:
    cols_to_process = [c for c in cols if c in out.columns]

  for col in cols_to_process:
    if out[col].dtype != "object" and not pd.api.types.is_string_dtype(out[col]):
      continue
    value_counts = out[col].value_counts(dropna=False)
    # Proportion of total rows (include NaN as a possible category)
    rare_vals = [
      val for val, count in value_counts.items()
      if (count / n_total) < min_prop
    ]
    binned = out[col].replace(rare_vals, "Other")
    out[col + suffix] = binned

  return out

def transform_skew(df, features=None, suffix="_skewfix"):
  """
  Reduce skew in numeric columns by choosing the transformation that minimizes |skewness|.
  Creates new columns with the given suffix; original columns and all rows preserved.

  features: None (all numeric non-boolean columns), a single column name (str), or list of names.
  Handles negatives, zeros, and missing values; uses shift for log/sqrt when needed.
  Tries: identity, Yeo-Johnson, log1p, sqrt, cube root; tie-break: first in that order.
  """
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import PowerTransformer

  out = df.copy()
  n_rows = len(out)
  if n_rows == 0:
    return out

  def is_numeric_nonboolean(ser):
    if not pd.api.types.is_numeric_dtype(ser):
      return False
    u = ser.dropna().unique()
    if len(u) <= 2 and set(np.asarray(u).astype(int)) <= {0, 1}:
      return False
    return True

  if features is None:
    cols_to_process = [c for c in out.columns if is_numeric_nonboolean(out[c])]
  elif isinstance(features, str):
    cols_to_process = [features] if features in out.columns and is_numeric_nonboolean(out[features]) else []
  else:
    cols_to_process = [c for c in features if c in out.columns and is_numeric_nonboolean(out[c])]

  for col in cols_to_process:
    s = out[col].astype("float64")
    valid = s.dropna()
    if len(valid) < 2:
      out[col + suffix] = s
      continue
    v_min, v_max = valid.min(), valid.max()
    eps = 1e-10

    def shifted_log1p(series):
      shift = max(0.0, -v_min) + eps
      out_vals = np.full(len(series), np.nan, dtype=float)
      mask = series.notna()
      out_vals[mask] = np.log1p(series.loc[mask].values + shift)
      return pd.Series(out_vals, index=series.index)

    def shifted_sqrt(series):
      shift = max(0.0, -v_min) + eps
      out_vals = np.full(len(series), np.nan, dtype=float)
      mask = series.notna()
      out_vals[mask] = np.sqrt(series.loc[mask].values + shift)
      return pd.Series(out_vals, index=series.index)

    def cbrt_series(series):
      out_vals = np.full(len(series), np.nan, dtype=float)
      mask = series.notna()
      out_vals[mask] = np.cbrt(series.loc[mask].values)
      return pd.Series(out_vals, index=series.index)

    def yeojohnson_series(series):
      valid_vals = series.dropna()
      if len(valid_vals) < 2:
        return series.copy()
      pt = PowerTransformer(method="yeo-johnson", standardize=False)
      try:
        pt.fit(valid_vals.values.reshape(-1, 1))
        out_vals = np.full(len(series), np.nan, dtype=float)
        out_vals[valid_vals.index] = pt.transform(valid_vals.values.reshape(-1, 1)).ravel()
        return pd.Series(out_vals, index=series.index)
      except Exception:
        return series.copy()

    candidates = [
      ("identity", s),
      ("yeojohnson", yeojohnson_series(s)),
      ("log1p", shifted_log1p(s)),
      ("sqrt", shifted_sqrt(s)),
      ("cbrt", cbrt_series(s)),
    ]
    best_name, best_series = None, None
    best_abs_skew = np.inf
    for name, transformed in candidates:
      t_clean = transformed.dropna()
      if len(t_clean) < 2:
        continue
      sk = t_clean.skew()
      if pd.isna(sk):
        continue
      abs_sk = abs(sk)
      if abs_sk < best_abs_skew:
        best_abs_skew = abs_sk
        best_name, best_series = name, transformed
    if best_series is None:
      out[col + suffix] = s
    else:
      out[col + suffix] = best_series

  return out

def impute_missing(df, features=None, group_cols=None):
  """
  Fill missing values using group-based then global fallback; no rows or columns dropped.
  Numeric: median (by group, then global). Categorical: mode (by group, then global).
  features: None (all columns with missing), single column name (str), or list of names.
  group_cols: None (auto-select *_clean categorical columns), or list of column names for grouping.
  Deterministic: same input produces same output.
  """
  import pandas as pd
  import numpy as np

  out = df.copy()
  n_rows = len(out)
  if n_rows == 0:
    return out

  # Columns that have at least one missing value
  has_missing = out.isna().sum()
  cols_with_missing = has_missing[has_missing > 0].index.tolist()

  if features is None:
    cols_to_impute = cols_with_missing
  elif isinstance(features, str):
    cols_to_impute = [features] if features in out.columns and features in cols_with_missing else []
  else:
    cols_to_impute = [c for c in features if c in out.columns]

  if not cols_to_impute:
    return out

  # Resolve grouping columns
  if group_cols is None:
    group_cols = [
      c for c in out.columns
      if (c.endswith("_clean") or c.endswith("_binned"))
      and (out[c].dtype == "object" or pd.api.types.is_string_dtype(out[c]))
      and out[c].nunique() <= 500
    ]
    if not group_cols:
      group_cols = []
  else:
    group_cols = [c for c in group_cols if c in out.columns]

  def group_fill_series(ser, global_fill, is_numeric):
    if ser.notna().any():
      if is_numeric:
        return ser.fillna(ser.median())
      mode_vals = ser.mode()
      if len(mode_vals) > 0:
        return ser.fillna(mode_vals.iloc[0])
    return ser.fillna(global_fill)

  for col in cols_to_impute:
    if out[col].isna().sum() == 0:
      continue
    is_numeric = pd.api.types.is_numeric_dtype(out[col])
    if is_numeric:
      global_fill = out[col].median()
      if pd.isna(global_fill):
        global_fill = 0
    else:
      mode_vals = out[col].mode()
      global_fill = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"

    if group_cols:
      out[col] = out.groupby(group_cols, dropna=False)[col].transform(
        lambda x: group_fill_series(x, global_fill, is_numeric)
      )
    out[col] = out[col].fillna(global_fill)

  return out

def cap_outliers_iqr(df, cols=None):
  """
  Cap (winsorize) extreme values using Tukey's IQR fence method. No rows or columns dropped.
  Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are set to the boundary.
  cols: None (all numeric non-boolean), single column name (str), or list of names.
  """
  import pandas as pd
  import numpy as np

  out = df.copy()
  if len(out) == 0:
    return out

  def is_numeric_nonboolean(ser):
    if not pd.api.types.is_numeric_dtype(ser):
      return False
    u = ser.dropna().unique()
    if len(u) <= 2 and set(np.asarray(u).astype(float).astype(int)) <= {0, 1}:
      return False
    return True

  if cols is None:
    cols_to_process = [c for c in out.columns if is_numeric_nonboolean(out[c])]
  elif isinstance(cols, str):
    cols_to_process = [cols] if cols in out.columns and is_numeric_nonboolean(out[cols]) else []
  else:
    cols_to_process = [c for c in cols if c in out.columns and is_numeric_nonboolean(out[c])]

  for col in cols_to_process:
    q1 = out[col].quantile(0.25)
    q3 = out[col].quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr):
      continue
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out[col] = out[col].clip(lower=lower, upper=upper)

  return out


def missing_data_diagnostics(df, missing_thresh=0.9, verbose=True):
  """
  Report missing data counts, proportions, and heuristic suggested mechanism (MCAR/MAR/MNAR).
  Does not modify the dataframe. MAR vs MNAR cannot be distinguished from data alone;
  the function reports 'MAR/MNAR' when missingness is associated with observed variables.
  """
  import pandas as pd
  import numpy as np
  from scipy import stats

  n_rows, n_cols = len(df), len(df.columns)
  if n_rows == 0 or n_cols == 0:
    if verbose:
      print("DataFrame is empty.")
    return {"summary": "empty", "cols_dropped": [], "rows_dropped": [], "per_column": {}}

  # Per-column missing counts and proportions
  missing_count = df.isna().sum()
  missing_prop = missing_count / n_rows
  cols_with_missing = missing_count[missing_count > 0].index.tolist()

  # Columns that would be dropped at missing_thresh (drop if proportion > thresh)
  cols_to_drop = missing_prop[missing_prop > missing_thresh].index.tolist()
  cols_after_drop = [c for c in df.columns if c not in cols_to_drop]
  n_cols_after = len(cols_after_drop)

  # Rows that would be dropped: use remaining columns only
  if n_cols_after > 0:
    row_missing_count = df[cols_after_drop].isna().sum(axis=1)
    row_missing_prop = row_missing_count / n_cols_after
    rows_to_drop_mask = row_missing_prop > missing_thresh
    rows_to_drop_count = rows_to_drop_mask.sum()
  else:
    rows_to_drop_count = n_rows

  # Heuristic mechanism per column with missingness
  alpha = 0.05
  per_column = {}
  for col in df.columns:
    n_miss = missing_count[col]
    if n_miss == 0:
      per_column[col] = {"missing_count": 0, "missing_prop": 0.0, "suggested_mechanism": "no missing"}
      continue
    prop = missing_prop[col]
    mechanism = "MCAR?"
    other_cols = [c for c in df.columns if c != col and df[c].notna().any()]
    for other in other_cols:
      if df[other].isna().all():
        continue
      mask_missing = df[col].isna()
      obs_missing = df.loc[mask_missing, other].dropna()
      obs_observed = df.loc[~mask_missing, other].dropna()
      if len(obs_missing) < 2 or len(obs_observed) < 2:
        continue
      if pd.api.types.is_numeric_dtype(df[other]):
        try:
          _, p = stats.ttest_ind(obs_missing, obs_observed, nan_policy="omit")
          if p is not None and not np.isnan(p) and p < alpha:
            mechanism = "MAR/MNAR"
            break
        except Exception:
          pass
      else:
        try:
          ctab = pd.crosstab(df[other].fillna("__NA__"), mask_missing.astype(int))
          if ctab.size >= 2 and ctab.shape[0] >= 1 and ctab.shape[1] == 2:
            _, p, _, _ = stats.chi2_contingency(ctab)
            if p is not None and p < alpha:
              mechanism = "MAR/MNAR"
              break
        except Exception:
          pass
    per_column[col] = {"missing_count": int(n_miss), "missing_prop": float(prop), "suggested_mechanism": mechanism}

  result = {
    "per_column": per_column,
    "cols_dropped": cols_to_drop,
    "rows_dropped_count": int(rows_to_drop_count) if n_cols_after > 0 else n_rows,
    "missing_thresh": missing_thresh,
  }

  if verbose:
    print("=== Missing data diagnostics ===")
    print(f"Threshold: drop if proportion missing > {missing_thresh}")
    print(f"Columns that would be dropped ({len(cols_to_drop)}): {cols_to_drop or 'none'}")
    print(f"Rows that would be dropped: {result['rows_dropped_count']}")
    if not cols_with_missing:
      print("No missing values in any column.")
    else:
      print("\nPer-column summary (columns with missing):")
      for col in cols_with_missing:
        info = per_column[col]
        print(f"  {col}: missing={info['missing_count']} ({info['missing_prop']:.2%}), suggested mechanism={info['suggested_mechanism']}")
    print("(MAR vs MNAR cannot be distinguished from data; 'MAR/MNAR' means missingness is associated with observed variables.)")

  return result

def missing_data_clean(df, missing_thresh=0.9, imputation_level='simple', diagnostics=False, missing_indicator=False):
  """
  Return a cleaned pandas DataFrame with all missing values appropriately filled in.
  (1) Drops columns/rows with proportion missing > missing_thresh;
  (2) Imputes remaining missing values (numeric: median/KNN/MICE; categorical: mode).
  Optionally prints diagnostics first or adds missing-indicator columns.
  """
  import pandas as pd
  import numpy as np
  from sklearn.impute import SimpleImputer, KNNImputer
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer

  out = df.copy()
  n_rows, n_cols = len(out), len(out.columns)
  if n_rows == 0 or n_cols == 0:
    return pd.DataFrame(out)

  if diagnostics:
    missing_data_diagnostics(out, missing_thresh=missing_thresh, verbose=True)

  # Drop columns with proportion missing > missing_thresh
  missing_per_col = out.isna().sum() / n_rows
  cols_drop = missing_per_col[missing_per_col > missing_thresh].index.tolist()
  out = out.drop(columns=cols_drop, errors="ignore")
  if len(out.columns) == 0:
    return pd.DataFrame(out)

  # Drop rows with proportion missing > missing_thresh (over remaining columns)
  n_c = len(out.columns)
  row_miss = out.isna().sum(axis=1) / n_c
  out = out.loc[row_miss <= missing_thresh].copy()

  if len(out) == 0:
    return pd.DataFrame(out)

  if not out.isna().any().any():
    return pd.DataFrame(out)

  # Optional missing indicators (add before imputation)
  if missing_indicator:
    cols_with_missing = [c for c in out.columns if out[c].isna().any()]
    for col in cols_with_missing:
      ind_name = f"{col}_was_missing"
      out[ind_name] = out[col].isna().astype(int)

  numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
  other_cols = [c for c in out.columns if c not in numeric_cols]

  # Impute numeric columns (assign back into DataFrame so index/columns are preserved)
  if numeric_cols:
    X_num = out[numeric_cols]
    if imputation_level == 'simple':
      imp_num = SimpleImputer(strategy='median')
      imputed = imp_num.fit_transform(X_num)
    elif imputation_level == 'knn':
      imp_num = KNNImputer()
      imputed = imp_num.fit_transform(X_num)
    elif imputation_level == 'mice':
      imp_num = IterativeImputer(max_iter=10, random_state=0)
      imputed = imp_num.fit_transform(X_num)
    else:
      imp_num = SimpleImputer(strategy='median')
      imputed = imp_num.fit_transform(X_num)
    out[numeric_cols] = pd.DataFrame(imputed, index=out.index, columns=numeric_cols)

  # Impute categorical / object columns with mode
  for col in other_cols:
    if out[col].isna().any():
      mode_val = out[col].mode()
      fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
      out[col] = out[col].fillna(fill_val)

  # Final pass: fill any remaining missing (e.g. datetime or other dtypes)
  for col in out.columns:
    if out[col].isna().any():
      if pd.api.types.is_numeric_dtype(out[col]):
        out[col] = out[col].fillna(out[col].median())
      else:
        mode_val = out[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
        out[col] = out[col].fillna(fill_val)

  return pd.DataFrame(out)

def manage_dates(df, columns=None, startdate=None, enddate=None, date_threshold=0.5):
  """
  Convert columns that are valid dates into datetime format and add new features:
  day, month, year, weekday, and hour (only if the column includes a time component).
  If startdate or enddate is specified, add columns with the number of days between
  that date and the date column value.
  """
  import pandas as pd

  out = df.copy()
  cols_to_check = list(columns) if columns is not None else list(out.columns)
  cols_to_check = [c for c in cols_to_check if c in out.columns]

  for col in cols_to_check:
    if pd.api.types.is_datetime64_any_dtype(out[col]):
      ser = out[col]
    elif pd.api.types.is_numeric_dtype(out[col]):
      continue
    else:
      parsed = pd.to_datetime(out[col], errors="coerce")
      non_null = out[col].notna()
      if non_null.sum() == 0:
        continue
      pct_valid = parsed.notna().loc[non_null].sum() / non_null.sum()
      if pct_valid < date_threshold:
        continue
      ser = parsed

    out[col] = ser
    out[f"{col}_day"] = ser.dt.day
    out[f"{col}_month"] = ser.dt.month
    out[f"{col}_year"] = ser.dt.year
    out[f"{col}_weekday"] = ser.dt.weekday
    has_time = (ser.dt.hour != 0) | (ser.dt.minute != 0) | (ser.dt.second != 0)
    if has_time.any():
      out[f"{col}_hour"] = ser.dt.hour
    if startdate is not None:
      start = pd.Timestamp(startdate)
      out[f"{col}_days_since_start"] = (ser - start).dt.days
    if enddate is not None:
      end = pd.Timestamp(enddate)
      out[f"{col}_days_until_end"] = (end - ser).dt.days

  return out


def _continuous_numeric_columns(df):
  """Return list of numeric column names that are not boolean-like (0/1 only)."""
  import pandas as pd
  out = []
  for col in df.columns:
    if df[col].dtype not in ("int64", "float64", "int32", "float32"):
      continue
    unique_vals = set(df[col].dropna().unique())
    if unique_vals.issubset({0, 1}):
      continue
    out.append(col)
  return out


def normalize(df, columns=None, drop_original=False, verbose=True):
  """
  For each selected numeric (non-boolean) continuous-like feature, compute skewness and
  apply the best normalizing transformation. Transformed values are added as new columns
  (suffix _normalized). Optionally drop the original columns.

  - Negative skew: tries yeojohnson, square, cube, exponent; picks the one that minimizes |skew|.
  - Positive skew: tries yeojohnson, sqrt, cbrt, ln; picks the one that minimizes |skew|.
  - If |skew| is already small, the column may be left unchanged (identity) or yeojohnson applied.

  Parameters
  ----------
  df : pandas.DataFrame
      Input dataframe (not modified in place; a copy is returned).
  columns : list of str or None
      Columns to process. If None, all acceptable numeric (non-boolean) columns are processed.
  drop_original : bool, default False
      If True, remove the original column after adding the normalized column.
  verbose : bool, default True
      If True, print per-column skew and chosen transformation.

  Returns
  -------
  pandas.DataFrame
      Copy of df with new _normalized columns (and optionally without originals).
  """
  import pandas as pd
  import numpy as np
  from scipy import stats

  out = df.copy()
  acceptable = _continuous_numeric_columns(out)
  if columns is None or len(columns) == 0:
    cols_to_process = acceptable
  else:
    cols_to_process = [c for c in columns if c in acceptable]
    missing = [c for c in columns if c not in acceptable]
    if missing and verbose:
      print(f"Columns skipped (not numeric continuous or not in df): {missing}")

  for col in cols_to_process:
    x = out[col].astype(float)
    # Drop NaN for skew and transform evaluation; we'll align by index when assigning
    valid = x.notna()
    x_clean = x.loc[valid]
    if len(x_clean) < 3:
      if verbose:
        print(f"{col}: skipped (too few non-null values)")
      continue

    skew_orig = x_clean.skew()
    if np.isnan(skew_orig):
      if verbose:
        print(f"{col}: skipped (skew is NaN)")
      continue

    best_skew = abs(skew_orig)
    best_name = "identity"
    best_vals = x.copy()

    def score_transform(vals, name):
      nonlocal best_skew, best_name, best_vals
      v_clean = vals.loc[valid].replace([np.inf, -np.inf], np.nan).dropna()
      if len(v_clean) < 3:
        return
      s = vals.loc[valid].skew()
      if np.isnan(s):
        return
      if abs(s) < best_skew:
        best_skew = abs(s)
        best_name = name
        best_vals = vals.copy()

    # Yeo-Johnson (handles negative and zero values)
    try:
      yj, _ = stats.yeojohnson(x_clean)
      yj_series = x.copy()
      yj_series.loc[valid] = yj
      score_transform(yj_series, "yeojohnson")
    except Exception:
      pass

    if skew_orig < 0:
      # Negative skew: square, cube, exponent
      sq = x ** 2
      score_transform(sq, "square")
      cb = x ** 3
      score_transform(cb, "cube")
      # Exponent can overflow; use clip or scaled exp if needed
      ex = np.exp(x)
      ex = ex.replace([np.inf, -np.inf], np.nan)
      if ex.notna().sum() >= 3:
        score_transform(ex, "exponent")
    else:
      # Positive skew: sqrt, cbrt, ln
      if (x_clean >= 0).all():
        sqrt_vals = np.sqrt(x)
        score_transform(sqrt_vals, "sqrt")
      if (x_clean >= 0).all():
        ln_vals = np.log1p(x)
        score_transform(ln_vals, "ln")
      cbrt_vals = np.cbrt(x)
      score_transform(cbrt_vals, "cbrt")

    new_col = f"{col}_normalized"
    out[new_col] = best_vals

    if drop_original:
      out = out.drop(columns=[col])

    if verbose:
      print(f"{col}: skew={skew_orig:.3f} -> best transform={best_name} (skew after={best_skew:.3f})")

  return pd.DataFrame(out)


def manage_outliers(df, epsilon=0.5, min_samples=5, action="nothing", report=True, exclude=None):
  """
  Detect outliers using DBSCAN clustering on numeric (non-boolean) continuous features,
  then optionally delete them, winsorize them, or do nothing. An `is_outlier` column
  (0/1) is always added so you can filter or inspect later.

  DBSCAN labels points as outlier (cluster -1) when they are not within epsilon
  distance of at least min_samples neighbors. Smaller epsilon = more restrictive
  (more points labeled as outliers); larger epsilon = less restrictive.

  Parameters
  ----------
  df : pandas.DataFrame
      Input dataframe (not modified in place; a copy is returned).
  epsilon : float, default 0.5
      Maximum distance between two samples for one to be in the neighborhood of the other.
      Features are standardized before clustering. Smaller = more restrictive.
  min_samples : int, default 5
      Minimum number of samples in a neighborhood for a point to be a core point.
  action : str, one of "nothing", "delete", "winsorize", default "nothing"
      - "nothing": only add `is_outlier`; do not change values or rows.
      - "delete": remove rows where is_outlier == 1.
      - "winsorize": for each numeric feature used in DBSCAN, clip outlier rows to
        the 1st and 99th percentile of inlier values (outliers only, per feature).
  report : bool, default True
      If True, print a summary: counts, epsilon/min_samples, and which features
      are most often extreme among outliers (by count of outliers with that feature
      beyond 2 standard deviations from the inlier mean).
  exclude : list of str or None, optional
      Column names to exclude from DBSCAN (and from winsorization). If None or
      empty, all eligible numeric continuous columns are used.

  Returns
  -------
  pandas.DataFrame
      Copy of df with `is_outlier` added; rows/values changed per `action`.
  """
  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import DBSCAN

  out = df.copy()
  feature_cols = _continuous_numeric_columns(out)
  # If exclude is provided, remove those columns from the feature set
  if exclude is not None and len(exclude) > 0:
    exclude_set = set(exclude)
    feature_cols = [c for c in feature_cols if c not in exclude_set]
  if not feature_cols:
    if report:
      print("No numeric (non-boolean) continuous columns found; returning df unchanged.")
    out["is_outlier"] = 0
    return pd.DataFrame(out)

  X = out[feature_cols].copy()
  for c in feature_cols:
    X[c] = X[c].fillna(out[c].median())
  X_arr = X.astype(float).values

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_arr)
  db = DBSCAN(eps=epsilon, min_samples=min_samples)
  labels = db.fit_predict(X_scaled)

  outlier_mask = labels == -1
  n_outliers = int(outlier_mask.sum())
  n_inliers = int((~outlier_mask).sum())
  n_clusters = len(set(labels) - {-1})

  out["is_outlier"] = (labels == -1).astype(int)

  if report:
    print("=== Outlier report (DBSCAN) ===")
    print(f"Epsilon: {epsilon}, min_samples: {min_samples}")
    print(f"Inliers: {n_inliers}, Outliers: {n_outliers}, Clusters: {n_clusters}")
    if n_outliers > 0:
      inlier_vals = X.loc[~outlier_mask]
      inlier_mean = inlier_vals.mean()
      inlier_std = inlier_vals.std().replace(0, np.nan)
      outlier_vals = X.loc[outlier_mask]
      extreme_count = {}
      for c in feature_cols:
        std_c = inlier_std[c]
        if pd.isna(std_c) or std_c == 0:
          extreme_count[c] = 0
          continue
        z = (outlier_vals[c] - inlier_mean[c]) / std_c
        extreme_count[c] = int((np.abs(z) > 2).sum())
      sorted_features = sorted(extreme_count.items(), key=lambda x: -x[1])
      print("Features most often extreme among outliers (|z| > 2 vs inlier mean):")
      for feat, count in sorted_features:
        print(f"  {feat}: {count} outlier(s)")

  if action == "delete" and n_outliers > 0:
    out = out.loc[~outlier_mask].copy()
  elif action == "winsorize" and n_outliers > 0:
    inlier_vals = out.loc[~outlier_mask, feature_cols]
    for c in feature_cols:
      low, high = np.nanpercentile(inlier_vals[c], [1, 99])
      out.loc[outlier_mask, c] = out.loc[outlier_mask, c].clip(low, high)

  return pd.DataFrame(out)

# =============================================================================
# EDA
# =============================================================================

def unistats(df):
  """
  Tabular statistics summary for every column.
  Numeric columns: count, unique, dtype, min, max, Q1, Q2, Q3, mean, median, mode, std, skew, kurt.
  Non-numeric: count, unique, dtype, mode only (dashes elsewhere).
  """
  import pandas as pd
  import numpy as np

  output_df = pd.DataFrame(columns=[
    'Count', 'Unique', 'Type',
    'Min', 'Max', '25%', '50%', '75%',
    'Mean', 'Median', 'Mode', 'Std', 'Skew', 'Kurt'
  ])
  for col in df.columns:
    count = df[col].count()
    unique = df[col].nunique()
    dtype = str(df[col].dtype)
    min_val = max_val = q1 = q2 = q3 = '-'
    mean_val = median_val = std_val = skew_val = kurt_val = '-'
    mode_series = df[col].mode()
    mode_val = mode_series.values[0] if len(mode_series) > 0 else '-'
    if pd.api.types.is_numeric_dtype(df[col]):
      min_val   = round(df[col].min(), 2)
      max_val   = round(df[col].max(), 2)
      q1        = round(df[col].quantile(0.25), 2)
      q2        = round(df[col].quantile(0.50), 2)
      q3        = round(df[col].quantile(0.75), 2)
      mean_val  = round(df[col].mean(), 2)
      median_val= round(df[col].median(), 2)
      if pd.api.types.is_numeric_dtype(pd.Series([mode_val])):
        mode_val = round(float(mode_val), 2)
      std_val   = round(df[col].std(), 2)
      skew_val  = round(df[col].skew(), 2)
      kurt_val  = round(df[col].kurt(), 2)
    output_df.loc[col] = (
      count, unique, dtype,
      min_val, max_val, q1, q2, q3,
      mean_val, median_val, mode_val, std_val, skew_val, kurt_val
    )
  return output_df


def bivariate(df, target):
  """
  Bivariate analysis of every feature against a target column.
  Numeric features: scatter plot + Pearson r.
  Categorical features: grouped boxplot (numeric target) or countplot (categorical target).
  Returns a DataFrame of correlation values for numeric features.
  """
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  corr_rows = []
  for col in df.columns:
    if col == target:
      continue
    if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
      r = df[[col, target]].dropna().corr().iloc[0, 1]
      corr_rows.append({"Feature": col, "Pearson_r": round(r, 4)})
      plt.figure(figsize=(7, 4))
      sns.scatterplot(data=df, x=col, y=target, alpha=0.4)
      plt.title(f"{col} vs {target}  (r={r:.3f})")
      plt.tight_layout()
      plt.show()
    elif not pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
      plt.figure(figsize=(9, 4))
      sns.boxplot(data=df, x=col, y=target)
      plt.title(f"{target} by {col}")
      plt.xticks(rotation=45, ha='right')
      plt.tight_layout()
      plt.show()
    elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[target]):
      plt.figure(figsize=(7, 4))
      sns.boxplot(data=df, x=target, y=col)
      plt.title(f"{col} by {target}")
      plt.xticks(rotation=45, ha='right')
      plt.tight_layout()
      plt.show()
    else:
      plt.figure(figsize=(9, 4))
      sns.countplot(data=df, x=col, hue=target)
      plt.title(f"{col} vs {target}")
      plt.xticks(rotation=45, ha='right')
      plt.tight_layout()
      plt.show()

  if corr_rows:
    return pd.DataFrame(corr_rows).sort_values("Pearson_r", key=abs, ascending=False).reset_index(drop=True)
  return pd.DataFrame()


def correlation_heatmap(df, figsize=(12, 10), annot=True, fmt='.2f', cmap='coolwarm'):
  """Plot a correlation heatmap for all numeric columns."""
  import matplotlib.pyplot as plt
  import seaborn as sns

  corr = df.select_dtypes('number').corr()
  plt.figure(figsize=figsize)
  sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5)
  plt.title('Correlation Heatmap')
  plt.tight_layout()
  plt.show()
  return corr


# =============================================================================
# DATA PREPARATION
# =============================================================================

def basic_wrangling(df, features=None, missing_threshold=0.95,
                    unique_threshold=0.95, messages=True):
  """
  Drop columns that are nearly all missing (>= missing_threshold) or nearly all
  unique (>= unique_threshold — likely ID columns). Returns modified copy.
  """
  import pandas as pd

  if features is None or len(features) == 0:
    features = list(df.columns)
  cols_to_drop = []
  for feat in features:
    if feat not in df.columns:
      continue
    rows = df.shape[0]
    if rows == 0:
      continue
    missing_pct = df[feat].isna().sum() / rows
    unique_pct  = df[feat].nunique() / rows
    if missing_pct >= missing_threshold:
      if messages:
        print(f"Dropping '{feat}': {missing_pct:.1%} missing")
      cols_to_drop.append(feat)
    elif unique_pct >= unique_threshold:
      if messages:
        print(f"Dropping '{feat}': {unique_pct:.1%} unique values (likely ID)")
      cols_to_drop.append(feat)
  return df.drop(columns=cols_to_drop, errors='ignore')


def missing_fill(df, target, mar='drop', large_dataset=200000, messages=True):
  """
  Impute missing values with MAR awareness.
  - mar='drop': drop rows where target is NaN.
  - mar='impute': impute target along with other features.
  Uses KNNImputer for large datasets, IterativeImputer for small ones.
  Returns a copy with no missing numeric values.
  """
  import pandas as pd
  from sklearn.impute import KNNImputer
  from sklearn.experimental import enable_iterative_imputer  # noqa
  from sklearn.impute import IterativeImputer

  df = df.copy()
  missing_target = df[target].isna().sum()
  if missing_target > 0:
    if mar == 'drop':
      if messages:
        print(f"Dropping {missing_target} rows missing target '{target}'")
      df = df[df[target].notna()].copy()
    else:
      if messages:
        print(f"Will impute {missing_target} missing values in target '{target}'")

  num_cols = df.select_dtypes(include='number').columns.tolist()
  missing_counts = df[num_cols].isna().sum()
  cols_missing = missing_counts[missing_counts > 0].index.tolist()
  if not cols_missing:
    if messages:
      print("No missing numeric values remaining.")
    return df

  if messages:
    print(f"Imputing {len(cols_missing)} numeric column(s): {cols_missing}")

  if df.shape[0] > large_dataset:
    imputer = KNNImputer()
  else:
    imputer = IterativeImputer(random_state=42, max_iter=10)

  df[num_cols] = imputer.fit_transform(df[num_cols])
  return df


def recode(df, col, mapping=None, conditions=None, choices=None,
           new_col=None, default='unknown'):
  """
  Recode a column using map/replace (mapping dict) or multi-condition logic
  (conditions + choices lists, like np.select). Creates new_col or overwrites col.

  Examples:
    recode(df, 'gender', mapping={'female': 0, 'male': 1})
    recode(df, 'age', conditions=[df['age']<18, df['age']<65],
           choices=['minor','adult'], default='senior', new_col='age_group')
  """
  import numpy as np

  out = df.copy()
  dest = new_col if new_col else col
  if mapping is not None:
    out[dest] = out[col].map(mapping)
  elif conditions is not None and choices is not None:
    out[dest] = np.select(conditions, choices, default=default)
  return out


def encode_features(df, cat_cols=None, strategy='onehot', drop_first=True):
  """
  Encode categorical columns.
  strategy='onehot' : pd.get_dummies (returns wide DataFrame).
  strategy='label'  : LabelEncoder per column (in-place on copy).
  """
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder

  out = df.copy()
  if cat_cols is None:
    cat_cols = out.select_dtypes(include='object').columns.tolist()

  if strategy == 'onehot':
    out = pd.get_dummies(out, columns=cat_cols, drop_first=drop_first)
  elif strategy == 'label':
    le = LabelEncoder()
    for col in cat_cols:
      if col in out.columns:
        out[col] = le.fit_transform(out[col].astype(str))
  return out


def scale_features(df, cols=None, strategy='standard'):
  """
  Scale numeric columns.
  strategy='standard' : zero mean, unit variance (StandardScaler).
  strategy='minmax'   : scale to [0, 1] (MinMaxScaler).
  Returns a copy with scaled columns replaced in-place.
  """
  import pandas as pd
  from sklearn.preprocessing import StandardScaler, MinMaxScaler

  out = df.copy()
  if cols is None:
    cols = out.select_dtypes(include='number').columns.tolist()

  scaler = StandardScaler() if strategy == 'standard' else MinMaxScaler()
  out[cols] = scaler.fit_transform(out[cols])
  return out


# =============================================================================
# MULTICOLLINEARITY
# =============================================================================

def compute_vif(X_df):
  """
  Compute Variance Inflation Factor for each column in X_df.
  Returns a DataFrame sorted by VIF descending. VIF > 10 suggests multicollinearity.
  """
  import pandas as pd
  from statsmodels.stats.outliers_influence import variance_inflation_factor

  vif_data = pd.DataFrame()
  vif_data["Feature"] = X_df.columns
  vif_data["VIF"] = [
    variance_inflation_factor(X_df.values.astype(float), i)
    for i in range(X_df.shape[1])
  ]
  return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def remove_high_vif(X_df, threshold=10.0):
  """
  Iteratively drop the highest-VIF feature until all VIF values are below threshold.
  Returns the reduced DataFrame.
  """
  import pandas as pd

  df = X_df.copy()
  while True:
    vif = compute_vif(df)
    max_vif = vif["VIF"].max()
    if max_vif > threshold:
      worst = vif.loc[vif["VIF"].idxmax(), "Feature"]
      print(f"Dropping '{worst}' (VIF={round(max_vif, 2)})")
      df = df.drop(columns=[worst])
    else:
      break
  return df


# =============================================================================
# MODELING PIPELINE BUILDER
# =============================================================================

def build_preprocessor(X_train):
  """
  Build a ColumnTransformer that median-imputes + standardizes numeric columns
  and mode-imputes + one-hot-encodes categorical columns.
  Returns (preprocessor, num_cols, cat_cols).
  """
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  from sklearn.impute import SimpleImputer

  num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
  cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

  numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
  ])
  categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
  ])
  preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
  ])
  return preprocessor, num_cols, cat_cols


def make_pipeline_for_model(X_train, model):
  """
  Wrap a scikit-learn estimator with the shared preprocessing pipeline.

  Calls build_preprocessor(X_train) to build a ColumnTransformer that
  median-imputes + scales numeric columns and mode-imputes + one-hot-encodes
  categorical columns, then combines it with `model` in a Pipeline so that
  all preprocessing is learned only on training data (no leakage).

  Parameters
  ----------
  X_train : pd.DataFrame
      Training feature matrix used to infer column types for the preprocessor.
  model : sklearn estimator
      Any unfitted classifier or regressor.

  Returns
  -------
  sklearn.pipeline.Pipeline
      Pipeline with steps [("preprocess", ColumnTransformer), ("model", estimator)].
  """
  from sklearn.pipeline import Pipeline

  preprocessor, _, _ = build_preprocessor(X_train)
  return Pipeline([("preprocess", preprocessor), ("model", model)])


def split_data(df, target, test_size=0.2, random_state=42, stratify=False):
  """
  Split df into X_train, X_test, y_train, y_test.
  stratify=True uses the target column for stratified splitting (classification).
  """
  from sklearn.model_selection import train_test_split

  X = df.drop(columns=[target])
  y = df[target]
  strat = y if stratify else None
  return train_test_split(X, y, test_size=test_size, random_state=random_state,
                          stratify=strat)


# =============================================================================
# MODEL EVALUATION — REGRESSION
# =============================================================================

def eval_regression(name, model, X_train, y_train, X_test, y_test, fit=True):
  """
  Fit (optional) and evaluate a regression model.
  Prints and returns a dict of MAE, RMSE, R2 vs a mean baseline.
  """
  import numpy as np
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

  if fit:
    model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  baseline = np.full_like(y_test, fill_value=float(y_train.mean()), dtype=float)
  b_mae  = mean_absolute_error(y_test, baseline)
  b_rmse = np.sqrt(mean_squared_error(y_test, baseline))

  mae  = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2   = r2_score(y_test, y_pred)

  print(f"\n{'='*60}")
  print(f"{name}")
  print(f"  Baseline MAE: {b_mae:.2f}  RMSE: {b_rmse:.2f}")
  print(f"  Model    MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")
  return {"model": name, "mae": mae, "rmse": rmse, "r2": r2,
          "baseline_mae": b_mae, "baseline_rmse": b_rmse}


# =============================================================================
# MODEL EVALUATION — CLASSIFICATION
# =============================================================================

def eval_classification(name, model, X_train, y_train, X_test, y_test, fit=True):
  """
  Fit (optional) and evaluate a classification model.
  Prints accuracy, log loss, and full classification report.
  Returns a dict of key metrics.
  """
  from sklearn.metrics import (accuracy_score, log_loss, classification_report,
                                roc_auc_score, f1_score)

  if fit:
    model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  acc = accuracy_score(y_test, y_pred)
  f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
  ll  = None
  auc = None

  if hasattr(model, 'predict_proba'):
    y_prob = model.predict_proba(X_test)
    ll = log_loss(y_test, y_prob)
    classes = sorted(y_test.unique())
    if len(classes) == 2:
      auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
      try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
      except Exception:
        pass

  print(f"\n{'='*60}")
  print(f"{name}")
  print(f"  Accuracy : {acc:.4f}")
  if ll  is not None: print(f"  Log Loss : {ll:.4f}")
  if auc is not None: print(f"  ROC AUC  : {auc:.4f}")
  print(classification_report(y_test, y_pred, digits=3, zero_division=0))
  return {"model": name, "accuracy": acc, "f1": f1, "log_loss": ll, "roc_auc": auc}


def plot_roc_curve(model, X_test, y_test, title='ROC Curve'):
  """Plot ROC curve for a binary classifier with predict_proba."""
  import matplotlib.pyplot as plt
  from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

  y_prob = model.predict_proba(X_test)[:, 1]
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  auc_val = roc_auc_score(y_test, y_prob)
  disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val)
  disp.plot()
  plt.title(f"{title} (AUC={auc_val:.4f})")
  plt.tight_layout()
  plt.show()
  return auc_val


def plot_confusion_matrix(model, X_test, y_test, labels=None, title='Confusion Matrix'):
  """Plot confusion matrix heatmap for a fitted classifier."""
  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

  y_pred = model.predict(X_test)
  cm = confusion_matrix(y_test, y_pred, labels=labels)
  disp = ConfusionMatrixDisplay(cm, display_labels=labels)
  disp.plot(values_format='d', cmap='Blues')
  plt.title(title)
  plt.tight_layout()
  plt.show()
  return cm


def plot_precision_recall(model, X_test, y_test, title='Precision-Recall Curve'):
  """Plot precision-recall curve for a binary classifier."""
  import matplotlib.pyplot as plt
  from sklearn.metrics import precision_recall_curve

  y_prob = model.predict_proba(X_test)[:, 1]
  precision, recall, _ = precision_recall_curve(y_test, y_prob)
  plt.figure(figsize=(7, 5))
  plt.plot(recall, precision, marker='.')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title(title)
  plt.tight_layout()
  plt.show()


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_model(model, X, y, cv=5, scoring='roc_auc',
                          stratified=True, random_state=42, verbose=True):
  """
  Run K-Fold cross-validation. Uses StratifiedKFold for classification scoring
  and KFold otherwise. Returns mean ± std of the chosen scoring metric.
  """
  import numpy as np
  from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

  if stratified:
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
  else:
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

  scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
  if verbose:
    print(f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}  (folds: {scores.round(4)})")
  return scores


def plot_learning_curve(model, X, y, cv=5, scoring='roc_auc',
                         train_sizes=None, random_state=42):
  """
  Plot learning curve to diagnose bias/variance. Shows train vs. validation
  score as training set size grows.
  """
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import learning_curve, StratifiedKFold

  if train_sizes is None:
    import numpy as np
    train_sizes = np.linspace(0.1, 1.0, 10)

  skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
  sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=train_sizes, cv=skf,
    scoring=scoring, n_jobs=-1
  )
  plt.figure(figsize=(9, 5))
  plt.plot(sizes, train_scores.mean(axis=1), label='Train')
  plt.fill_between(sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
  plt.plot(sizes, val_scores.mean(axis=1), label='Validation')
  plt.fill_between(sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
  plt.xlabel('Training Set Size')
  plt.ylabel(scoring)
  plt.title('Learning Curve')
  plt.legend()
  plt.tight_layout()
  plt.show()


def plot_validation_curve(model, X, y, param_name, param_range,
                           cv=5, scoring='roc_auc', random_state=42):
  """
  Plot validation curve showing how a single hyperparameter affects train vs.
  validation performance.
  """
  import matplotlib.pyplot as plt
  from sklearn.model_selection import validation_curve, StratifiedKFold

  skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
  train_scores, val_scores = validation_curve(
    model, X, y,
    param_name=param_name, param_range=param_range,
    cv=skf, scoring=scoring, n_jobs=-1
  )
  plt.figure(figsize=(9, 5))
  plt.plot(param_range, train_scores.mean(axis=1), label='Train')
  plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
  plt.xlabel(param_name)
  plt.ylabel(scoring)
  plt.title(f'Validation Curve — {param_name}')
  plt.legend()
  plt.tight_layout()
  plt.show()


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_grid(model, param_grid, X_train, y_train, cv=5,
              scoring='roc_auc', random_state=42, verbose=True):
  """
  Run GridSearchCV. Returns the best estimator and prints best params + score.
  """
  from sklearn.model_selection import GridSearchCV, StratifiedKFold

  skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
  gs = GridSearchCV(model, param_grid, cv=skf, scoring=scoring, n_jobs=-1)
  gs.fit(X_train, y_train)
  if verbose:
    print(f"Best params : {gs.best_params_}")
    print(f"Best CV {scoring}: {gs.best_score_:.4f}")
  return gs.best_estimator_, gs


def tune_random(model, param_dist, X_train, y_train, n_iter=50, cv=5,
                scoring='roc_auc', random_state=42, verbose=True):
  """
  Run RandomizedSearchCV. Returns the best estimator and prints best params + score.
  """
  from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

  skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
  rs = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=skf,
                          scoring=scoring, n_jobs=-1, random_state=random_state)
  rs.fit(X_train, y_train)
  if verbose:
    print(f"Best params : {rs.best_params_}")
    print(f"Best CV {scoring}: {rs.best_score_:.4f}")
  return rs.best_estimator_, rs


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_features_filter(X_train_arr, y_train, feature_names, k=10,
                            method='anova'):
  """
  Filter-based feature selection.
  method='anova' : ANOVA F-test (linear relationships, classification).
  method='mi'    : Mutual information (non-linear, classification).
  Returns (selector, selected_feature_names).
  """
  import numpy as np
  from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

  score_func = f_classif if method == 'anova' else mutual_info_classif
  selector = SelectKBest(score_func=score_func, k=k)
  selector.fit(X_train_arr, y_train)
  mask = selector.get_support()
  selected = list(np.array(feature_names)[mask])
  print(f"Selected {len(selected)} features ({method}): {selected}")
  return selector, selected


def select_features_rfe(X_train_arr, y_train, feature_names,
                         estimator=None, cv=5, scoring='roc_auc'):
  """
  Recursive Feature Elimination with Cross-Validation (RFECV).
  Returns (rfecv, selected_feature_names).
  """
  import numpy as np
  from sklearn.feature_selection import RFECV
  from sklearn.model_selection import StratifiedKFold
  from sklearn.linear_model import LogisticRegression

  if estimator is None:
    estimator = LogisticRegression(max_iter=1000)
  rfecv = RFECV(estimator=estimator, step=1,
                cv=StratifiedKFold(cv), scoring=scoring, n_jobs=1)
  rfecv.fit(X_train_arr, y_train)
  selected = list(np.array(feature_names)[rfecv.support_])
  print(f"Optimal features: {rfecv.n_features_} — {selected}")
  return rfecv, selected


def permutation_importance_report(model, X_test_arr, y_test, feature_names,
                                   n_repeats=10, scoring='roc_auc',
                                   random_state=42, top_n=15):
  """
  Compute and plot permutation feature importance.
  Returns a sorted DataFrame of mean importance ± std.
  """
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.inspection import permutation_importance

  result = permutation_importance(model, X_test_arr, y_test,
                                  n_repeats=n_repeats,
                                  scoring=scoring,
                                  random_state=random_state,
                                  n_jobs=1)
  pfi = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': result.importances_mean,
    'Importance Std':  result.importances_std
  }).sort_values('Importance Mean', ascending=False).reset_index(drop=True)

  top = pfi.head(top_n)
  plt.figure(figsize=(9, 6))
  plt.barh(top['Feature'][::-1], top['Importance Mean'][::-1],
           xerr=top['Importance Std'][::-1])
  plt.xlabel(f'Mean Decrease in {scoring}')
  plt.title('Permutation Feature Importance')
  plt.tight_layout()
  plt.show()
  return pfi


def feature_importance_plot(model_step, feature_names, top_n=20, title='Feature Importance'):
  """
  Bar chart of MDI feature importances from a tree-based model step
  (e.g. pipeline.named_steps['rf']).
  """
  import pandas as pd
  import matplotlib.pyplot as plt

  importances = pd.Series(
    model_step.feature_importances_, index=feature_names
  ).sort_values(ascending=False).head(top_n)

  plt.figure(figsize=(9, 6))
  plt.barh(importances.index[::-1], importances.values[::-1])
  plt.xlabel('MDI Importance')
  plt.title(title)
  plt.tight_layout()
  plt.show()
  return importances


def plot_logit_coefficients(logit_pipe, top_n=20, title='Logistic Regression Coefficients'):
  """
  Horizontal bar chart of the top logistic regression coefficients by |coef|.

  Bars are colored by direction: positive coefficients in steelblue, negative
  in salmon, with a vertical line at zero. Useful for a quick causal-style
  interpretation of which features most strongly push the log-odds up or down.

  Parameters
  ----------
  logit_pipe : sklearn.pipeline.Pipeline
      A fitted Pipeline whose steps are named "preprocess" (ColumnTransformer)
      and "model" (LogisticRegression).
  top_n : int, default 20
      Number of top features (by |coef|) to display.
  title : str
      Plot title.

  Returns
  -------
  pd.DataFrame
      DataFrame with columns ['feature', 'coef', 'abs_coef'] sorted by abs_coef
      descending (all features, not just top_n).
  """
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  preprocessor = logit_pipe.named_steps["preprocess"]
  logit_model   = logit_pipe.named_steps["model"]

  feature_names = preprocessor.get_feature_names_out()
  coefs = logit_model.coef_.reshape(-1)

  coef_df = pd.DataFrame({
    "feature":  feature_names,
    "coef":     coefs,
    "abs_coef": np.abs(coefs),
  }).sort_values("abs_coef", ascending=False)

  top = coef_df.head(top_n).iloc[::-1]
  colors = ["steelblue" if c > 0 else "salmon" for c in top["coef"]]

  plt.figure(figsize=(10, 7))
  plt.barh(top["feature"], top["coef"], color=colors)
  plt.axvline(0, color="black", linewidth=1)
  plt.title(title)
  plt.xlabel("Coefficient (log-odds per 1 unit of preprocessed feature)")
  plt.tight_layout()
  plt.show()

  return coef_df


# =============================================================================
# OLS / CAUSAL MODELING
# =============================================================================

def ols_summary(df, target, add_const=True, encode_cats=True):
  """
  Fit a statsmodels OLS model and print the full summary.
  Returns the fitted results object.
  """
  import pandas as pd
  import numpy as np
  import statsmodels.api as sm

  y = df[target]
  X = df.drop(columns=[target])

  if encode_cats:
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    if cat_cols:
      X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
  X = X.select_dtypes(include='number')
  bool_cols = X.select_dtypes(include='bool').columns
  if len(bool_cols):
    X[bool_cols] = X[bool_cols].astype(int)
  if add_const:
    X = sm.add_constant(X)

  model = sm.OLS(y, X).fit()
  print(model.summary())
  return model


# =============================================================================
# TIME SERIES
# =============================================================================

def decompose_time_series(series, model='additive', period=12):
  """
  Seasonal decomposition using statsmodels. Plots trend, seasonal, and residual.
  Returns the DecomposeResult object.
  """
  import matplotlib.pyplot as plt
  from statsmodels.tsa.seasonal import seasonal_decompose

  result = seasonal_decompose(series.dropna(), model=model, period=period)
  result.plot()
  plt.tight_layout()
  plt.show()
  return result


def fit_arima(train_y, order=(1, 1, 1), seasonal_order=None,
              exog_train=None, disp=False):
  """
  Fit ARIMA or SARIMA(X) model.
  seasonal_order: tuple (P,D,Q,s) — if provided, fits SARIMAX.
  Returns fitted model result.
  """
  from statsmodels.tsa.arima.model import ARIMA
  from statsmodels.tsa.statespace.sarimax import SARIMAX

  if seasonal_order is not None:
    model = SARIMAX(train_y, exog=exog_train, order=order,
                    seasonal_order=seasonal_order)
  else:
    model = ARIMA(train_y, exog=exog_train, order=order)
  result = model.fit(disp=disp)
  print(f"AIC: {result.aic:.2f}  BIC: {result.bic:.2f}")
  return result


def evaluate_time_series(df_forecast, test_y, model_col):
  """
  Compute MAE and RMSE for a single forecast column aligned to test_y.
  Returns dict with model name, MAE, RMSE.
  """
  import numpy as np
  import pandas as pd

  yhat = df_forecast.loc[test_y.index, model_col]
  err  = (test_y - yhat).dropna()
  mae  = float(np.mean(np.abs(err)))
  rmse = float(np.sqrt(np.mean(err ** 2)))
  print(f"{model_col}  MAE={mae:.4f}  RMSE={rmse:.4f}")
  return {"model": model_col, "mae": mae, "rmse": rmse}


# =============================================================================
# NLP — TEXT FEATURES
# =============================================================================

def extract_text_features(df, text_col):
  """
  Extract POS counts and named-entity counts from a text column using spaCy.
  Adds columns: Nouns, Verbs, Adjectives, Numbers, Pronouns,
                People, Organizations, Locations, Dates, Times.
  Returns a copy of df with new columns.
  """
  import spacy

  nlp = spacy.load('en_core_web_sm')
  docs = list(nlp.pipe(df[text_col].astype(str)))

  def count_pos(doc, pos):
    return sum(1 for t in doc if t.pos_ == pos)

  def count_ent(doc, label):
    return sum(1 for e in doc.ents if e.label_ == label)

  out = df.copy()
  out['Nouns']         = [count_pos(d, 'NOUN')    for d in docs]
  out['Verbs']         = [count_pos(d, 'VERB')    for d in docs]
  out['Adjectives']    = [count_pos(d, 'ADJ')     for d in docs]
  out['Numbers']       = [count_pos(d, 'NUM')     for d in docs]
  out['Pronouns']      = [count_pos(d, 'PRON')    for d in docs]
  out['People']        = [count_ent(d, 'PERSON')  for d in docs]
  out['Organizations'] = [count_ent(d, 'ORG')     for d in docs]
  out['Locations']     = [count_ent(d, 'GPE')     for d in docs]
  out['Dates']         = [count_ent(d, 'DATE')    for d in docs]
  out['Times']         = [count_ent(d, 'TIME')    for d in docs]
  return out


def add_sentiment(df, text_col):
  """
  Add VADER sentiment scores (pos, neg, neu, compound) for a text column.
  Returns a copy of df with new sentiment columns.
  """
  import nltk
  from nltk.sentiment import SentimentIntensityAnalyzer

  nltk.download('vader_lexicon', quiet=True)
  sia = SentimentIntensityAnalyzer()

  out = df.copy()
  scores = out[text_col].astype(str).apply(sia.polarity_scores)
  out['sentiment_pos']      = scores.apply(lambda s: s['pos'])
  out['sentiment_neg']      = scores.apply(lambda s: s['neg'])
  out['sentiment_neu']      = scores.apply(lambda s: s['neu'])
  out['sentiment_compound'] = scores.apply(lambda s: s['compound'])
  return out


def get_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
  """
  Return mean-pooled sentence embeddings as a numpy array using HuggingFace.
  Each row corresponds to one text in the input list.
  """
  import torch
  from transformers import AutoTokenizer, AutoModel

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model     = AutoModel.from_pretrained(model_name)
  inputs    = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
  with torch.no_grad():
    outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).numpy()


# =============================================================================
# MONITORING / DRIFT DETECTION
# =============================================================================

def compute_psi(reference, current, bins=10):
  """
  Compute Population Stability Index (PSI) between reference and current distributions.
  PSI < 0.10: stable. 0.10-0.25: moderate drift. > 0.25: significant drift.
  """
  import numpy as np

  breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
  breakpoints[-1] += 1e-9
  ref_counts = np.histogram(reference, bins=breakpoints)[0]
  cur_counts = np.histogram(current,   bins=breakpoints)[0]
  eps = 1e-4
  ref_pct = ref_counts / len(reference) + eps
  cur_pct = cur_counts / len(current)   + eps
  psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
  label = ('stable' if psi < 0.10 else
           'moderate drift' if psi < 0.25 else 'significant drift')
  print(f"PSI = {psi:.4f} ({label})")
  return round(psi, 4)


def check_categorical_drift(reference, current, feature_name):
  """
  Compare category proportions between reference and current Series.
  Warns about new categories. Returns a DataFrame with ref %, current %, shift.
  """
  import pandas as pd

  new_cats = set(current.unique()) - set(reference.unique())
  if new_cats:
    print(f"WARNING: New categories in '{feature_name}': {new_cats}")
  ref_dist = reference.value_counts(normalize=True).sort_index()
  cur_dist = current.value_counts(normalize=True).sort_index()
  comp = pd.DataFrame({'reference': ref_dist, 'current': cur_dist}).fillna(0)
  comp['shift'] = (comp['current'] - comp['reference']).round(3)
  return comp


def plot_drift(reference, current, feature_name):
  """
  Overlay normalized histograms of training vs. current feature distributions.
  """
  import matplotlib.pyplot as plt

  plt.figure(figsize=(8, 4))
  plt.hist(reference, bins=30, alpha=0.6, label='Training',
           density=True, color='steelblue')
  plt.hist(current,   bins=30, alpha=0.5, label='Current',
           density=True, color='salmon')
  plt.title(f'Distribution Comparison: {feature_name}')
  plt.xlabel(feature_name)
  plt.ylabel('Density')
  plt.legend()
  plt.tight_layout()
  plt.show()


def monitor_all_features(X_train, X_current, psi_bins=10, psi_threshold=0.25):
  """
  Run PSI on every numeric feature and categorical drift on every object feature.
  Prints a summary and returns a DataFrame of PSI values sorted descending.
  """
  import pandas as pd

  rows = []
  for col in X_train.columns:
    if col not in X_current.columns:
      continue
    if pd.api.types.is_numeric_dtype(X_train[col]):
      psi_val = compute_psi(X_train[col].dropna().values,
                             X_current[col].dropna().values,
                             bins=psi_bins)
      flag = psi_val > psi_threshold
      rows.append({'feature': col, 'psi': psi_val, 'drift_flag': flag})
    else:
      drift = check_categorical_drift(X_train[col], X_current[col], col)
      max_shift = drift['shift'].abs().max()
      rows.append({'feature': col, 'psi': None,
                   'max_cat_shift': round(max_shift, 3),
                   'drift_flag': max_shift > 0.1})
  return pd.DataFrame(rows).sort_values('psi', ascending=False, na_position='last')


# =============================================================================
# DEPLOYMENT / MLOPS
# =============================================================================

def save_model(model, path='model.pkl'):
  """Serialize a fitted model/pipeline to disk with joblib."""
  import joblib
  from pathlib import Path
  joblib.dump(model, Path(path))
  print(f"Model saved to {path}")


def load_model(path='model.pkl'):
  """Load a joblib-serialized model/pipeline from disk."""
  import joblib
  from pathlib import Path
  model = joblib.load(Path(path))
  print(f"Model loaded from {path}")
  return model


def save_metrics(metrics_dict, path='metrics.json'):
  """Save a metrics dictionary to JSON."""
  import json
  from pathlib import Path
  with open(Path(path), 'w') as f:
    json.dump(metrics_dict, f, indent=2)
  print(f"Metrics saved to {path}")


def load_metrics(path='metrics.json'):
  """Load a metrics dictionary from JSON."""
  import json
  from pathlib import Path
  with open(Path(path)) as f:
    return json.load(f)


def should_retrain(metrics_path='metrics.json', metric='f1', threshold=0.70):
  """
  Return True if the saved model metric falls below the threshold,
  indicating retraining is needed.
  """
  import json
  from pathlib import Path
  try:
    with open(Path(metrics_path)) as f:
      m = json.load(f)
    score = m[metric]
    if score < threshold:
      print(f"{metric}={score:.4f} below {threshold}. Retraining needed.")
      return True
    print(f"{metric}={score:.4f} OK. No retraining needed.")
    return False
  except FileNotFoundError:
    print("No metrics file found. Retraining needed.")
    return True


def champion_challenger(new_metrics, champion_metrics_path,
                        new_model_path, champion_model_path, metric='f1'):
  """
  Compare challenger vs. champion on a single metric.
  Promotes challenger (copies to champion_model_path) if it wins.
  Returns True if challenger was promoted, False otherwise.
  """
  import json, shutil
  from pathlib import Path

  try:
    with open(Path(champion_metrics_path)) as f:
      champ = json.load(f)
    champ_score = champ[metric]
  except FileNotFoundError:
    print("No champion found. Promoting new model.")
    shutil.copy(new_model_path, champion_model_path)
    return True

  chal_score = new_metrics[metric]
  print(f"Champion {metric}: {champ_score:.4f}  |  Challenger {metric}: {chal_score:.4f}")
  if chal_score > champ_score:
    print("Challenger wins. Promoting new model.")
    shutil.copy(new_model_path, champion_model_path)
    return True
  print("Champion retains.")
  return False


def load_to_warehouse(df, table_name, db_path='warehouse.db', if_exists='replace'):
  """Write a DataFrame to a SQLite warehouse table."""
  import sqlite3
  from pathlib import Path
  conn = sqlite3.connect(str(Path(db_path)))
  df.to_sql(table_name, conn, if_exists=if_exists, index=False)
  conn.commit()
  conn.close()
  print(f"Loaded {len(df)} rows to '{table_name}' in {db_path}")


def read_from_warehouse(table_name, db_path='warehouse.db'):
  """Read a table from a SQLite warehouse into a DataFrame."""
  import sqlite3
  import pandas as pd
  from pathlib import Path
  conn = sqlite3.connect(str(Path(db_path)))
  df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
  conn.close()
  return df


def log_metrics(metrics_dict, model_version, feature_list,
                db_path='warehouse.db'):
  """
  Append training metrics to a metrics_log table in SQLite.
  Creates the table if it doesn't exist.
  """
  import sqlite3, json
  from datetime import datetime
  from pathlib import Path

  conn = sqlite3.connect(str(Path(db_path)))
  cur = conn.cursor()
  cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics_log (
      log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
      trained_at   TEXT NOT NULL,
      model_version TEXT NOT NULL,
      accuracy     REAL, f1 REAL, roc_auc REAL,
      row_count_train INTEGER, row_count_test INTEGER,
      features     TEXT
    )""")
  cur.execute("""
    INSERT INTO metrics_log
      (trained_at, model_version, accuracy, f1, roc_auc,
       row_count_train, row_count_test, features)
    VALUES (?,?,?,?,?,?,?,?)""", (
    datetime.utcnow().isoformat(),
    model_version,
    metrics_dict.get('accuracy'),
    metrics_dict.get('f1'),
    metrics_dict.get('roc_auc'),
    metrics_dict.get('row_count_train'),
    metrics_dict.get('row_count_test'),
    json.dumps(feature_list)
  ))
  conn.commit()
  conn.close()
  print(f"Metrics logged for '{model_version}'.")


def register_model(model_version, model_path, metrics_dict,
                   feature_list, notes='', db_path='warehouse.db'):
  """Record a model version in the model_registry SQLite table."""
  import sqlite3, json
  from datetime import datetime
  from pathlib import Path

  conn = sqlite3.connect(str(Path(db_path)))
  cur = conn.cursor()
  cur.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
      registry_id   INTEGER PRIMARY KEY AUTOINCREMENT,
      model_version TEXT NOT NULL,
      model_path    TEXT NOT NULL,
      trained_at    TEXT NOT NULL,
      accuracy      REAL, f1 REAL, roc_auc REAL,
      features      TEXT, notes TEXT,
      is_active     INTEGER DEFAULT 0
    )""")
  cur.execute("""
    INSERT INTO model_registry
      (model_version, model_path, trained_at, accuracy, f1, roc_auc,
       features, notes, is_active)
    VALUES (?,?,?,?,?,?,?,?,0)""", (
    model_version, str(model_path),
    datetime.utcnow().isoformat(),
    metrics_dict.get('accuracy'),
    metrics_dict.get('f1'),
    metrics_dict.get('roc_auc'),
    json.dumps(feature_list), notes
  ))
  conn.commit()
  conn.close()
  print(f"Model '{model_version}' registered.")


def promote_model(model_version, db_path='warehouse.db'):
  """Set a single model version as active in the model_registry."""
  import sqlite3
  from pathlib import Path

  conn = sqlite3.connect(str(Path(db_path)))
  cur = conn.cursor()
  cur.execute("UPDATE model_registry SET is_active = 0")
  cur.execute("UPDATE model_registry SET is_active = 1 WHERE model_version = ?",
              (model_version,))
  conn.commit()
  conn.close()
  print(f"Model '{model_version}' promoted to active.")


def plot_metrics_history(db_path='warehouse.db', threshold=0.70):
  """
  Plot accuracy, F1, and ROC AUC over time from metrics_log in SQLite.
  Draws a horizontal alert threshold line.
  """
  import sqlite3
  import pandas as pd
  import matplotlib.pyplot as plt
  from pathlib import Path

  conn = sqlite3.connect(str(Path(db_path)))
  history = pd.read_sql("SELECT * FROM metrics_log ORDER BY trained_at", conn)
  conn.close()

  fig, ax = plt.subplots(figsize=(11, 5))
  for col, marker in [('accuracy', 'o'), ('f1', 's'), ('roc_auc', '^')]:
    if col in history.columns:
      ax.plot(history['trained_at'], history[col], marker=marker, label=col)
  ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1,
             label=f'Alert threshold ({threshold})')
  ax.set_xlabel('Training Date')
  ax.set_ylabel('Metric Value')
  ax.set_title('Model Performance Over Time')
  plt.xticks(rotation=45)
  plt.legend()
  plt.tight_layout()
  plt.show()


