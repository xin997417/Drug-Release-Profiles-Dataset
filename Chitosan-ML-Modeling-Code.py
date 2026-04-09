import os
import json
import time
import warnings
import itertools
import traceback
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

file_path = r""
output_dir = r""
os.makedirs(output_dir, exist_ok=True)

random_state = 42
target_col = "release"
group_col = "Number"

candidate_features = [
    "CR",
    "CS_conc",
    "DD",
    "SA/V",
    "DL",
    "Drug_Mw",
    "Drug_Tm",
    "Drug_pKa",
    "pH",
    "Time"
]

analysis_schemes = {
    "unconstrained": {
        "mandatory_features": [],
        "feature_num_range": list(range(2, len(candidate_features) + 1)),
        "description": "Unconstrained search (2-10)"
    },
    "core_constrained": {
        "mandatory_features": ["CS_conc", "DD", "SA/V", "Time"],
        "feature_num_range": list(range(4, len(candidate_features) + 1)),  
        "description": "Core mechanism-constrained search (CS_conc, DD, SA/V, Time fixed)"
    },
    "all_material_constrained": {
        "mandatory_features": ["CR", "CS_conc", "DD", "SA/V", "DL", "Time"],
        "feature_num_range": list(range(6, len(candidate_features) + 1)),  
        "description": "All material-property constrained search (CR, CS_conc, DD, SA/V, DL, Time fixed)"
    }
}

RUN_STAGE2_TUNING = True

top_n_candidates_per_k = 1

include_model_global_best_subset = True

n_splits_inner = 5

n_outer_repeats = 5
outer_test_size = 0.2

screen_models = {
    "LightGBM": lgb.LGBMRegressor(
        random_state=random_state,
        verbose=-1,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8
    ),
    "RandomForest": RandomForestRegressor(
        random_state=random_state,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        verbose=0,
        random_state=random_state,
        depth=4,
        learning_rate=0.05,
        n_estimators=300,
        l2_leaf_reg=3
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3
    )
}

tune_models = {
    "LightGBM": (
        lgb.LGBMRegressor(
            random_state=random_state,
            verbose=-1
        ),
        {
            "n_estimators": [300, 500],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [15, 31],
            "min_child_samples": [5, 10],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        }
    ),

    "RandomForest": (
        RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1
        ),
        {
            "n_estimators": [300, 500],
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [1, 2, 4]
        }
    ),

    "XGBoost": (
        xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        ),
        {
            "n_estimators": [300, 500],
            "learning_rate": [0.03, 0.05],
            "max_depth": [3, 4],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_lambda": [1, 3]
        }
    ),

    "CatBoost": (
        CatBoostRegressor(
            verbose=0,
            random_state=random_state
        ),
        {
            "depth": [4, 6],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [300, 500],
            "l2_leaf_reg": [3, 5]
        }
    ),

    "GradientBoosting": (
        GradientBoostingRegressor(
            random_state=random_state
        ),
        {
            "n_estimators": [300, 500],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3]
        }
    ),

    "MLP": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                random_state=random_state,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
        {
            "model__hidden_layer_sizes": [(32,), (64,), (128,)],
            "model__alpha": [1e-4, 1e-3],
            "model__learning_rate_init": [5e-4, 1e-3]
        }
    ),

    "DNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                random_state=random_state,
                max_iter=1800,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
        {
            "model__hidden_layer_sizes": [(128, 64), (128, 64, 32), (256, 128, 64)],
            "model__alpha": [1e-4, 1e-3],
            "model__learning_rate_init": [3e-4, 5e-4]
        }
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    ),

    "AdaBoost": (
        AdaBoostRegressor(
            random_state=random_state
        ),
        {
            "n_estimators": [100, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "loss": ["linear", "square"]
        }
    ),

    "KernelRidge": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KernelRidge())
        ]),
        {
            "model__alpha": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "laplacian"],
            "model__gamma": [0.01, 0.1, 1.0]
        }
    )
}

def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def safe_json_dumps(x):
    return json.dumps(to_builtin(x), ensure_ascii=False)

def safe_json_loads(x):
    return json.loads(x)

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    ae_array = np.abs(y_true - y_pred)

    return {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "ae_mean": float(np.mean(ae_array)),
        "ae_array": ae_array
    }

def ordered_subset(all_features, subset_features):
    subset_set = set(subset_features)
    return [f for f in all_features if f in subset_set]

def generate_subsets(all_features, mandatory_features, feature_num_range):
    for f in mandatory_features:
        if f not in all_features:
            raise ValueError(f"Mandatory feature {f} is not included in the candidate feature list.")

    optional_features = [f for f in all_features if f not in mandatory_features]
    all_subsets = []

    for total_k in feature_num_range:
        if total_k < len(mandatory_features):
            continue

        optional_k = total_k - len(mandatory_features)
        combos = itertools.combinations(optional_features, optional_k)

        for combo in combos:
            subset = ordered_subset(all_features, list(mandatory_features) + list(combo))
            all_subsets.append(subset)

    return all_subsets

def get_groupkfold_splits(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups))

def fit_model(X_train, y_train, model_template, params=None):
    if params is None:
        params = {}
    model = clone(model_template)
    model.set_params(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model_on_splits(X, y, groups, features, model_template, splits, params=None):
    if params is None:
        params = {}

    fold_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr = X.iloc[tr_idx][features]
        X_va = X.iloc[va_idx][features]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        try:
            model = fit_model(
                X_train=X_tr,
                y_train=y_tr,
                model_template=model_template,
                params=params
            )

            pred = model.predict(X_va)
            metric = calc_metrics(y_va, pred)

            fold_rows.append({
                "fold": fold_id,
                "r2": metric["r2"],
                "mse": metric["mse"],
                "rmse": metric["rmse"],
                "mae": metric["mae"],
                "mape": metric["mape"]
            })
        except Exception as e:
            print(f"[Evaluation Error] fold={fold_id}, features={features}, params={params}, error={e}")
            traceback.print_exc()
            fold_rows.append({
                "fold": fold_id,
                "r2": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan
            })

    fold_df = pd.DataFrame(fold_rows)
    return {
        "r2_mean": fold_df["r2"].mean(),
        "r2_std": fold_df["r2"].std(ddof=0),
        "mse_mean": fold_df["mse"].mean(),
        "mse_std": fold_df["mse"].std(ddof=0),
        "rmse_mean": fold_df["rmse"].mean(),
        "rmse_std": fold_df["rmse"].std(ddof=0),
        "mae_mean": fold_df["mae"].mean(),
        "mae_std": fold_df["mae"].std(ddof=0),
        "mape_mean": fold_df["mape"].mean(),
        "mape_std": fold_df["mape"].std(ddof=0),
        "fold_detail_df": fold_df
    }

def evaluate_subset_screening(X, y, groups, subset, models, splits):
    row = {
        "feature_num": len(subset),
        "features": safe_json_dumps(subset)
    }

    mean_r2_list = []
    mean_rmse_list = []

    for model_name, model_template in models.items():
        eval_result = evaluate_model_on_splits(
            X=X,
            y=y,
            groups=groups,
            features=subset,
            model_template=model_template,
            splits=splits,
            params=None
        )

        row[f"{model_name}_r2_mean"] = eval_result["r2_mean"]
        row[f"{model_name}_r2_std"] = eval_result["r2_std"]
        row[f"{model_name}_mse_mean"] = eval_result["mse_mean"]
        row[f"{model_name}_mse_std"] = eval_result["mse_std"]
        row[f"{model_name}_rmse_mean"] = eval_result["rmse_mean"]
        row[f"{model_name}_rmse_std"] = eval_result["rmse_std"]
        row[f"{model_name}_mae_mean"] = eval_result["mae_mean"]
        row[f"{model_name}_mae_std"] = eval_result["mae_std"]
        row[f"{model_name}_mape_mean"] = eval_result["mape_mean"]
        row[f"{model_name}_mape_std"] = eval_result["mape_std"]

        mean_r2_list.append(eval_result["r2_mean"])
        mean_rmse_list.append(eval_result["rmse_mean"])

    row["mean_r2_across_models"] = np.mean(mean_r2_list)
    row["mean_rmse_across_models"] = np.mean(mean_rmse_list)

    return row

def tune_model_with_groupkfold(X, y, groups, features, model_name, model_template, param_grid, n_splits=5):
    inner_splits = get_groupkfold_splits(X, y, groups, n_splits=n_splits)
    param_list = list(ParameterGrid(param_grid)) if param_grid else [{}]

    best_params = None
    best_r2_mean = -np.inf
    best_rmse_mean = np.inf
    best_r2_std = np.inf

    tuning_rows = []

    print(f"\nStart tuning -> Model: {model_name} | Feature count: {len(features)} | Parameter combinations: {len(param_list)}")

    for i, params in enumerate(param_list, 1):
        eval_result = evaluate_model_on_splits(
            X=X,
            y=y,
            groups=groups,
            features=features,
            model_template=model_template,
            splits=inner_splits,
            params=params
        )

        tuning_rows.append({
            "model": model_name,
            "features": safe_json_dumps(features),
            "feature_num": len(features),
            "params": safe_json_dumps(params),
            "cv_r2_mean": eval_result["r2_mean"],
            "cv_r2_std": eval_result["r2_std"],
            "cv_mse_mean": eval_result["mse_mean"],
            "cv_mse_std": eval_result["mse_std"],
            "cv_rmse_mean": eval_result["rmse_mean"],
            "cv_rmse_std": eval_result["rmse_std"],
            "cv_mae_mean": eval_result["mae_mean"],
            "cv_mae_std": eval_result["mae_std"],
            "cv_mape_mean": eval_result["mape_mean"],
            "cv_mape_std": eval_result["mape_std"]
        })

        is_better = False
        if eval_result["r2_mean"] > best_r2_mean:
            is_better = True
        elif np.isclose(eval_result["r2_mean"], best_r2_mean, atol=1e-8):
            if eval_result["rmse_mean"] < best_rmse_mean:
                is_better = True
            elif np.isclose(eval_result["rmse_mean"], best_rmse_mean, atol=1e-8) and eval_result["r2_std"] < best_r2_std:
                is_better = True

        if is_better:
            best_params = deepcopy(params)
            best_r2_mean = eval_result["r2_mean"]
            best_rmse_mean = eval_result["rmse_mean"]
            best_r2_std = eval_result["r2_std"]

        if i % 5 == 0 or i == len(param_list):
            print(
                f"  Progress: {i}/{len(param_list)} | "
                f"Current best CV R2={best_r2_mean:.4f} | RMSE={best_rmse_mean:.4f}"
            )

    tuning_df = pd.DataFrame(tuning_rows)
    best_row = tuning_df.sort_values(
        by=["cv_r2_mean", "cv_rmse_mean", "cv_r2_std"],
        ascending=[False, True, True]
    ).iloc[0].to_dict()

    return best_params, best_row, tuning_df

def build_best_per_count_per_model(screen_df, model_names):
    rows = []

    for model_name in model_names:
        rmse_col = f"{model_name}_rmse_mean"
        r2_col = f"{model_name}_r2_mean"
        rmse_std_col = f"{model_name}_rmse_std"
        r2_std_col = f"{model_name}_r2_std"

        for feature_num in sorted(screen_df["feature_num"].unique()):
            sub_df = screen_df[screen_df["feature_num"] == feature_num].copy()

            best_row = sub_df.sort_values(
                by=[rmse_col, r2_col, rmse_std_col],
                ascending=[True, False, True]
            ).iloc[0]

            rows.append({
                "model": model_name,
                "feature_num": int(feature_num),
                "features": best_row["features"],
                "rmse_mean": best_row[rmse_col],
                "rmse_std": best_row[rmse_std_col],
                "r2_mean": best_row[r2_col],
                "r2_std": best_row[r2_std_col]
            })

    return pd.DataFrame(rows)

def build_best_per_count_across_models(screen_df, model_names):
    tmp = screen_df.copy()

    for model_name in model_names:
        tmp[f"{model_name}_rmse_rank"] = tmp[f"{model_name}_rmse_mean"].rank(
            ascending=True,
            method="average"
        )
        tmp[f"{model_name}_r2_rank"] = tmp[f"{model_name}_r2_mean"].rank(
            ascending=False,
            method="average"
        )

    tmp["mean_rmse_rank"] = tmp[[f"{m}_rmse_rank" for m in model_names]].mean(axis=1)
    tmp["mean_r2_rank"] = tmp[[f"{m}_r2_rank" for m in model_names]].mean(axis=1)

    rows = []
    for feature_num in sorted(tmp["feature_num"].unique()):
        sub_df = tmp[tmp["feature_num"] == feature_num].copy()

        best_row = sub_df.sort_values(
            by=["mean_rmse_rank", "mean_rmse_across_models", "mean_r2_across_models"],
            ascending=[True, True, False]
        ).iloc[0]

        rows.append({
            "feature_num": int(feature_num),
            "features": best_row["features"],
            "mean_rmse_across_models": best_row["mean_rmse_across_models"],
            "mean_r2_across_models": best_row["mean_r2_across_models"],
            "mean_rmse_rank": best_row["mean_rmse_rank"],
            "mean_r2_rank": best_row["mean_r2_rank"]
        })

    return pd.DataFrame(rows)

def plot_rmse_vs_feature_count(best_per_count_per_model_df, scheme_name, scheme_desc, save_path):
    marker_map = {
        "LightGBM": "o",
        "RandomForest": "s",
        "XGBoost": "D",
        "CatBoost": "^",
        "GradientBoosting": "P"
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    global_best_row = best_per_count_per_model_df.sort_values(
        by=["rmse_mean", "r2_mean", "rmse_std"],
        ascending=[True, False, True]
    ).iloc[0]

    for model_name in best_per_count_per_model_df["model"].unique():
        sub = best_per_count_per_model_df[best_per_count_per_model_df["model"] == model_name].copy()
        sub = sub.sort_values("feature_num")

        ax.plot(
            sub["feature_num"],
            sub["rmse_mean"],
            marker=marker_map.get(model_name, "o"),
            linewidth=2,
            markersize=8,
            label=model_name
        )

    ax.scatter(
        [global_best_row["feature_num"]],
        [global_best_row["rmse_mean"]],
        marker="*",
        s=350,
        label="Global best"
    )

    best_features = safe_json_loads(global_best_row["features"])
    annotation_text = (
        f"Best: {global_best_row['model']}\n"
        f"k={int(global_best_row['feature_num'])}\n"
        f"RMSE={global_best_row['rmse_mean']:.3f}\n"
        f"{', '.join(best_features)}"
    )

    ax.annotate(
        annotation_text,
        xy=(global_best_row["feature_num"], global_best_row["rmse_mean"]),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95),
        arrowprops=dict(arrowstyle="->", lw=1.2)
    )

    ax.set_title(f"Model Error vs Number of Features ({scheme_desc})", fontsize=18, fontweight="bold")
    ax.set_xlabel("Number of features", fontsize=14)
    ax.set_ylabel("Model error (RMSE)", fontsize=14)
    ax.set_xticks(sorted(best_per_count_per_model_df["feature_num"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_scheme_best_rmse_comparison(scheme_summary_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    for scheme_name in scheme_summary_df["scheme_name"].unique():
        sub = scheme_summary_df[scheme_summary_df["scheme_name"] == scheme_name].copy()
        sub = sub.sort_values("feature_num")
        ax.plot(
            sub["feature_num"],
            sub["best_rmse_across_models"],
            marker="o",
            linewidth=2,
            markersize=7,
            label=scheme_name
        )

    ax.set_title("Best RMSE Across Schemes by Feature Count", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of features", fontsize=13)
    ax.set_ylabel("Best RMSE across models", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def build_prediction_rows(df_part, y_true, y_pred, split_name, scheme_name, candidate_id, model_name, feature_num, features, outer_repeat, best_params):
    out = df_part.copy()
    out[target_col] = np.asarray(y_true)
    out["pred"] = np.asarray(y_pred)
    out["residual"] = out[target_col] - out["pred"]
    out["AE"] = np.abs(out["residual"])

    denom = np.maximum(np.abs(out[target_col].astype(float)), 1e-8)
    out["APE"] = np.abs(out["residual"] / denom) * 100.0

    out["split"] = split_name
    out["scheme_name"] = scheme_name
    out["candidate_id"] = candidate_id
    out["model"] = model_name
    out["feature_num"] = feature_num
    out["features"] = safe_json_dumps(features)
    out["outer_repeat"] = outer_repeat
    out["best_params_this_repeat"] = safe_json_dumps(best_params)

    return out

def build_metric_row(split_name, scheme_name, candidate_id, model_name, feature_num, features, outer_repeat, best_params, metric_dict):
    return {
        "split": split_name,
        "scheme_name": scheme_name,
        "candidate_id": candidate_id,
        "model": model_name,
        "feature_num": feature_num,
        "features": safe_json_dumps(features),
        "outer_repeat": outer_repeat,
        "best_params_this_repeat": safe_json_dumps(best_params),
        "r2": metric_dict["r2"],
        "mse": metric_dict["mse"],
        "rmse": metric_dict["rmse"],
        "mae": metric_dict["mae"],
        "ae_mean": metric_dict["ae_mean"],
        "mape": metric_dict["mape"]
    }

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

if group_col not in df.columns:
    raise ValueError(f"Group column not found in the dataset: {group_col}")
if target_col not in df.columns:
    raise ValueError(f"Target column not found in the dataset: {target_col}")

missing_features = [c for c in candidate_features if c not in df.columns]
if missing_features:
    raise ValueError(f"The following candidate features are missing from the dataset: {missing_features}")

df[group_col] = df[group_col].ffill()
df = df.dropna(subset=[group_col]).copy()
df = df.dropna(subset=[target_col]).copy()
df[group_col] = df[group_col].astype(str)

df = df[[group_col] + candidate_features + [target_col]].copy()

before_drop = len(df)
df = df.dropna(subset=candidate_features).copy()
after_drop = len(df)

print("=" * 100)
print("Data loading completed")
print(f"Total number of data points: {len(df)}")
print(f"Number of curves: {df[group_col].nunique()}")
print(f"Number of data points removed due to missing features: {before_drop - after_drop}")
print("=" * 100)

X_all = df[candidate_features].copy()
y_all = df[target_col].copy()
groups_all = df[group_col].copy()

if df[group_col].nunique() < n_splits_inner:
    raise ValueError(f"The number of curves is insufficient for {n_splits_inner}-fold GroupKFold.")

print("10 candidate features:")
print(candidate_features)

screen_splits = get_groupkfold_splits(X_all, y_all, groups_all, n_splits=n_splits_inner)

all_scheme_global_best_rows = []
all_scheme_feature_count_summary_rows = []
candidate_pool_rows = []

for scheme_name, scheme_info in analysis_schemes.items():
    scheme_dir = os.path.join(output_dir, scheme_name)
    os.makedirs(scheme_dir, exist_ok=True)

    mandatory_features = scheme_info["mandatory_features"]
    feature_num_range = scheme_info["feature_num_range"]
    scheme_desc = scheme_info["description"]

    print("\n" + "=" * 100)
    print(f"Start scheme: {scheme_name}")
    print(f"Description: {scheme_desc}")
    print(f"Mandatory features: {mandatory_features if mandatory_features else 'None'}")
    print(f"Feature count search range: {feature_num_range}")
    print("=" * 100)

    subsets = generate_subsets(
        all_features=candidate_features,
        mandatory_features=mandatory_features,
        feature_num_range=feature_num_range
    )

    print(f"Number of feature subsets to evaluate in this scheme: {len(subsets)}")

    screen_rows = []
    start_time = time.time()

    for idx, subset in enumerate(subsets, 1):
        row = evaluate_subset_screening(
            X=X_all,
            y=y_all,
            groups=groups_all,
            subset=subset,
            models=screen_models,
            splits=screen_splits
        )
        row["scheme_name"] = scheme_name
        row["scheme_description"] = scheme_desc
        row["mandatory_features"] = safe_json_dumps(mandatory_features)
        screen_rows.append(row)

        if idx % 20 == 0 or idx == len(subsets):
            elapsed = (time.time() - start_time) / 60
            print(f"[{scheme_name}] Progress: {idx}/{len(subsets)} | Elapsed time: {elapsed:.2f} min")

    screen_df = pd.DataFrame(screen_rows)

    screen_save_path = os.path.join(scheme_dir, f"stage1_{scheme_name}_all_subset_screening.csv")
    screen_df.to_csv(screen_save_path, index=False, encoding="utf-8-sig")
    print(f"All screening results saved to: {screen_save_path}")

    best_per_count_per_model_df = build_best_per_count_per_model(
        screen_df=screen_df,
        model_names=list(screen_models.keys())
    )
    best_per_count_per_model_df["scheme_name"] = scheme_name
    best_per_count_per_model_df["scheme_description"] = scheme_desc

    best_per_count_per_model_save_path = os.path.join(
        scheme_dir,
        f"stage1_{scheme_name}_best_per_count_per_model.csv"
    )
    best_per_count_per_model_df.to_csv(best_per_count_per_model_save_path, index=False, encoding="utf-8-sig")

    best_per_count_across_models_df = build_best_per_count_across_models(
        screen_df=screen_df,
        model_names=list(screen_models.keys())
    )
    best_per_count_across_models_df["scheme_name"] = scheme_name
    best_per_count_across_models_df["scheme_description"] = scheme_desc

    best_per_count_across_models_save_path = os.path.join(
        scheme_dir,
        f"stage1_{scheme_name}_best_per_count_across_models.csv"
    )
    best_per_count_across_models_df.to_csv(best_per_count_across_models_save_path, index=False, encoding="utf-8-sig")

    rmse_plot_save_path = os.path.join(scheme_dir, f"stage1_{scheme_name}_rmse_vs_feature_count.png")
    plot_rmse_vs_feature_count(
        best_per_count_per_model_df=best_per_count_per_model_df,
        scheme_name=scheme_name,
        scheme_desc=scheme_desc,
        save_path=rmse_plot_save_path
    )
    print(f"RMSE plot saved to: {rmse_plot_save_path}")

    scheme_global_best_row = best_per_count_per_model_df.sort_values(
        by=["rmse_mean", "r2_mean", "rmse_std"],
        ascending=[True, False, True]
    ).iloc[0].to_dict()

    all_scheme_global_best_rows.append({
        "scheme_name": scheme_name,
        "scheme_description": scheme_desc,
        "best_model": scheme_global_best_row["model"],
        "best_feature_num": int(scheme_global_best_row["feature_num"]),
        "best_features": scheme_global_best_row["features"],
        "best_rmse_mean": scheme_global_best_row["rmse_mean"],
        "best_rmse_std": scheme_global_best_row["rmse_std"],
        "best_r2_mean": scheme_global_best_row["r2_mean"],
        "best_r2_std": scheme_global_best_row["r2_std"]
    })

    for feature_num in sorted(best_per_count_per_model_df["feature_num"].unique()):
        sub = best_per_count_per_model_df[best_per_count_per_model_df["feature_num"] == feature_num].copy()
        best_sub = sub.sort_values(by=["rmse_mean", "r2_mean"], ascending=[True, False]).iloc[0]

        all_scheme_feature_count_summary_rows.append({
            "scheme_name": scheme_name,
            "scheme_description": scheme_desc,
            "feature_num": int(feature_num),
            "best_model_at_this_k": best_sub["model"],
            "best_features_at_this_k": best_sub["features"],
            "best_rmse_across_models": best_sub["rmse_mean"],
            "best_r2_across_models": best_sub["r2_mean"]
        })

    tmp = screen_df.copy()

    for model_name in screen_models.keys():
        tmp[f"{model_name}_rmse_rank"] = tmp[f"{model_name}_rmse_mean"].rank(
            ascending=True,
            method="average"
        )

    tmp["mean_rmse_rank"] = tmp[[f"{m}_rmse_rank" for m in screen_models.keys()]].mean(axis=1)

    for k in sorted(tmp["feature_num"].unique()):
        sub_k = tmp[tmp["feature_num"] == k].copy()
        sub_k = sub_k.sort_values(
            by=["mean_rmse_rank", "mean_rmse_across_models", "mean_r2_across_models"],
            ascending=[True, True, False]
        ).head(top_n_candidates_per_k)

        for _, row in sub_k.iterrows():
            candidate_pool_rows.append({
                "scheme_name": scheme_name,
                "scheme_description": scheme_desc,
                "feature_num": int(row["feature_num"]),
                "features": row["features"],
                "candidate_source": "best_across_models_by_k"
            })

    if include_model_global_best_subset:
        for model_name in screen_models.keys():
            sub_model = screen_df.sort_values(
                by=[f"{model_name}_rmse_mean", f"{model_name}_r2_mean", f"{model_name}_rmse_std"],
                ascending=[True, False, True]
            ).iloc[0]

            candidate_pool_rows.append({
                "scheme_name": scheme_name,
                "scheme_description": scheme_desc,
                "feature_num": int(sub_model["feature_num"]),
                "features": sub_model["features"],
                "candidate_source": f"global_best_of_{model_name}"
            })

    print(f"\n[{scheme_name}] Global best within this scheme:")
    print(f"Model: {scheme_global_best_row['model']}")
    print(f"Feature count: {int(scheme_global_best_row['feature_num'])}")
    print(f"Feature subset: {safe_json_loads(scheme_global_best_row['features'])}")
    print(f"RMSE: {scheme_global_best_row['rmse_mean']:.4f}")
    print(f"R2: {scheme_global_best_row['r2_mean']:.4f}")

scheme_global_best_df = pd.DataFrame(all_scheme_global_best_rows)
scheme_global_best_df = scheme_global_best_df.sort_values(
    by=["best_rmse_mean", "best_r2_mean"],
    ascending=[True, False]
).reset_index(drop=True)

scheme_global_best_save_path = os.path.join(output_dir, "stage1_all_schemes_global_best_summary.csv")
scheme_global_best_df.to_csv(scheme_global_best_save_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("Global best summary across all three schemes")
print("=" * 100)
print(scheme_global_best_df)

scheme_feature_count_summary_df = pd.DataFrame(all_scheme_feature_count_summary_rows)
scheme_feature_count_summary_save_path = os.path.join(output_dir, "stage1_all_schemes_feature_count_summary.csv")
scheme_feature_count_summary_df.to_csv(scheme_feature_count_summary_save_path, index=False, encoding="utf-8-sig")

scheme_compare_plot_path = os.path.join(output_dir, "stage1_compare_best_rmse_across_schemes.png")
plot_scheme_best_rmse_comparison(scheme_feature_count_summary_df, scheme_compare_plot_path)

print(f"\nCross-scheme summary table saved to: {scheme_feature_count_summary_save_path}")
print(f"Cross-scheme comparison plot saved to: {scheme_compare_plot_path}")

candidate_pool_df = pd.DataFrame(candidate_pool_rows)
candidate_pool_df = candidate_pool_df.drop_duplicates(subset=["scheme_name", "features"]).reset_index(drop=True)
candidate_pool_df["candidate_id"] = range(1, len(candidate_pool_df) + 1)

candidate_pool_save_path = os.path.join(output_dir, "stage2_candidate_pool_from_all_schemes.csv")
candidate_pool_df.to_csv(candidate_pool_save_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("Stage 2 candidate pool")
print("=" * 100)
print(candidate_pool_df.head(30))
print(f"\nCandidate pool saved to: {candidate_pool_save_path}")

if RUN_STAGE2_TUNING:
    print("\n" + "=" * 100)
    print("Stage 2: Candidate pool × tuned models")
    print("Repeated Group Holdout (outer loop) + GroupKFold tuning (inner loop)")
    print("=" * 100)

    outer_eval_summary_rows = []
    outer_eval_detail_rows = []
    all_tuning_rows = []

    pred_detail_rows = []
    metric_detail_rows = []

    gss = GroupShuffleSplit(
        n_splits=n_outer_repeats,
        test_size=outer_test_size,
        random_state=random_state
    )

    diag_dir = os.path.join(output_dir, "stage2_model_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    for _, c_row in candidate_pool_df.iterrows():
        candidate_id = int(c_row["candidate_id"])
        scheme_name = c_row["scheme_name"]
        scheme_description = c_row["scheme_description"]
        features = safe_json_loads(c_row["features"])

        print("\n" + "-" * 100)
        print(f"Current candidate candidate_id={candidate_id}")
        print(f"Source scheme: {scheme_name}")
        print(f"Description: {scheme_description}")
        print(f"Feature count: {len(features)}")
        print(f"Feature subset: {features}")
        print("-" * 100)

        for model_name, (model_template, param_grid) in tune_models.items():
            print("\n" + "~" * 60)
            print(f"Model: {model_name}")
            print("~" * 60)

            repeat_metric_rows = []

            for repeat_id, (tr_idx, va_idx) in enumerate(gss.split(X_all, y_all, groups_all), 1):
                X_tr_full = X_all.iloc[tr_idx].copy()
                y_tr = y_all.iloc[tr_idx].copy()
                groups_tr = groups_all.iloc[tr_idx].copy()

                X_va_full = X_all.iloc[va_idx].copy()
                y_va = y_all.iloc[va_idx].copy()
                groups_va = groups_all.iloc[va_idx].copy()

                print(
                    f"\n[Outer Repeat {repeat_id}/{n_outer_repeats}] "
                    f"Training curves={groups_tr.nunique()} | Validation curves={groups_va.nunique()}"
                )

                try:
                    best_params, best_row, tuning_df = tune_model_with_groupkfold(
                        X=X_tr_full,
                        y=y_tr,
                        groups=groups_tr,
                        features=features,
                        model_name=model_name,
                        model_template=model_template,
                        param_grid=param_grid,
                        n_splits=n_splits_inner
                    )

                    tuning_df["candidate_id"] = candidate_id
                    tuning_df["scheme_name"] = scheme_name
                    tuning_df["scheme_description"] = scheme_description
                    tuning_df["outer_repeat"] = repeat_id
                    all_tuning_rows.append(tuning_df)

                    
                    best_model = fit_model(
                        X_train=X_tr_full[features],
                        y_train=y_tr,
                        model_template=model_template,
                        params=best_params
                    )

                    
                    pred_tr = best_model.predict(X_tr_full[features])
                    metric_tr = calc_metrics(y_tr, pred_tr)

                    
                    pred_va = best_model.predict(X_va_full[features])
                    metric_va = calc_metrics(y_va, pred_va)

                    
                    repeat_metric_rows.append({
                        "candidate_id": candidate_id,
                        "scheme_name": scheme_name,
                        "scheme_description": scheme_description,
                        "model": model_name,
                        "feature_num": len(features),
                        "features": safe_json_dumps(features),
                        "outer_repeat": repeat_id,
                        "val_r2": metric_va["r2"],
                        "val_rmse": metric_va["rmse"],
                        "val_mae": metric_va["mae"],
                        "best_params_this_repeat": safe_json_dumps(best_params),
                        "inner_cv_r2_mean": best_row["cv_r2_mean"],
                        "inner_cv_r2_std": best_row["cv_r2_std"],
                        "inner_cv_rmse_mean": best_row["cv_rmse_mean"],
                        "inner_cv_rmse_std": best_row["cv_rmse_std"]
                    })

                    
                    tr_pred_df = build_prediction_rows(
                        df_part=df.iloc[tr_idx].copy(),
                        y_true=y_tr.values,
                        y_pred=pred_tr,
                        split_name="train",
                        scheme_name=scheme_name,
                        candidate_id=candidate_id,
                        model_name=model_name,
                        feature_num=len(features),
                        features=features,
                        outer_repeat=repeat_id,
                        best_params=best_params
                    )
                    va_pred_df = build_prediction_rows(
                        df_part=df.iloc[va_idx].copy(),
                        y_true=y_va.values,
                        y_pred=pred_va,
                        split_name="test",
                        scheme_name=scheme_name,
                        candidate_id=candidate_id,
                        model_name=model_name,
                        feature_num=len(features),
                        features=features,
                        outer_repeat=repeat_id,
                        best_params=best_params
                    )

                    pred_detail_rows.append(tr_pred_df)
                    pred_detail_rows.append(va_pred_df)

                    
                    metric_detail_rows.append(build_metric_row(
                        split_name="train",
                        scheme_name=scheme_name,
                        candidate_id=candidate_id,
                        model_name=model_name,
                        feature_num=len(features),
                        features=features,
                        outer_repeat=repeat_id,
                        best_params=best_params,
                        metric_dict=metric_tr
                    ))
                    metric_detail_rows.append(build_metric_row(
                        split_name="test",
                        scheme_name=scheme_name,
                        candidate_id=candidate_id,
                        model_name=model_name,
                        feature_num=len(features),
                        features=features,
                        outer_repeat=repeat_id,
                        best_params=best_params,
                        metric_dict=metric_va
                    ))

                    print(
                        f"[Outer Repeat Result] {model_name} | "
                        f"Train RMSE={metric_tr['rmse']:.4f} | Test RMSE={metric_va['rmse']:.4f}"
                    )

                except Exception as e:
                    print(f"[Outer Repeat Error] candidate_id={candidate_id}, model={model_name}, repeat={repeat_id}, error={e}")
                    traceback.print_exc()

            if len(repeat_metric_rows) == 0:
                continue

            repeat_metric_df = pd.DataFrame(repeat_metric_rows)
            outer_eval_detail_rows.append(repeat_metric_df)

            outer_eval_summary_rows.append({
                "candidate_id": candidate_id,
                "scheme_name": scheme_name,
                "scheme_description": scheme_description,
                "model": model_name,
                "feature_num": len(features),
                "features": safe_json_dumps(features),
                "outer_r2_mean": repeat_metric_df["val_r2"].mean(),
                "outer_r2_std": repeat_metric_df["val_r2"].std(ddof=0),
                "outer_rmse_mean": repeat_metric_df["val_rmse"].mean(),
                "outer_rmse_std": repeat_metric_df["val_rmse"].std(ddof=0),
                "outer_mae_mean": repeat_metric_df["val_mae"].mean(),
                "outer_mae_std": repeat_metric_df["val_mae"].std(ddof=0)
            })

    
    if all_tuning_rows:
        tuning_all_df = pd.concat(all_tuning_rows, ignore_index=True)
        tuning_save_path = os.path.join(output_dir, "stage2_all_tuning_records_all_schemes.csv")
        tuning_all_df.to_csv(tuning_save_path, index=False, encoding="utf-8-sig")
        print(f"\nAll tuning records saved to: {tuning_save_path}")
    else:
        tuning_all_df = pd.DataFrame()
        tuning_save_path = None
        print("\n[Notice] No tuning records were saved successfully.")

    if outer_eval_detail_rows:
        outer_eval_detail_df = pd.concat(outer_eval_detail_rows, ignore_index=True)
        outer_detail_save_path = os.path.join(output_dir, "stage2_outer_repeat_detail_all_schemes.csv")
        outer_eval_detail_df.to_csv(outer_detail_save_path, index=False, encoding="utf-8-sig")
        print(f"Outer repeat details saved to: {outer_detail_save_path}")
    else:
        outer_eval_detail_df = pd.DataFrame()
        outer_detail_save_path = None
        print("[Notice] No outer repeat details were saved successfully.")

    if len(outer_eval_summary_rows) == 0:
        raise RuntimeError("No valid results were obtained in Stage 2. Please check the error messages.")

    outer_eval_summary_df = pd.DataFrame(outer_eval_summary_rows)
    outer_eval_summary_df = outer_eval_summary_df.sort_values(
        by=["outer_r2_mean", "outer_rmse_mean", "outer_r2_std"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    outer_summary_save_path = os.path.join(output_dir, "stage2_outer_repeat_summary_all_schemes.csv")
    outer_eval_summary_df.to_csv(outer_summary_save_path, index=False, encoding="utf-8-sig")
    print(f"Outer repeat summary saved to: {outer_summary_save_path}")

    print("\n" + "=" * 100)
    print("Stage 2 development summary results (top 15)")
    print("=" * 100)
    print(outer_eval_summary_df.head(15))

    
    if pred_detail_rows:
        pred_detail_df = pd.concat(pred_detail_rows, ignore_index=True)
        pred_detail_save_path = os.path.join(output_dir, "stage2_prediction_detail_all_rows.csv")
        pred_detail_df.to_csv(pred_detail_save_path, index=False, encoding="utf-8-sig")
        print(f"Point-wise prediction details saved to: {pred_detail_save_path}")

        
        ae_boxplot_df = pred_detail_df[[
            "split", "scheme_name", "candidate_id", "model", "feature_num",
            group_col, "AE"
        ]].copy()
        ae_boxplot_path = os.path.join(output_dir, "stage2_AE_for_boxplot.csv")
        ae_boxplot_df.to_csv(ae_boxplot_path, index=False, encoding="utf-8-sig")
        print(f"AE boxplot data saved to: {ae_boxplot_path}")
    else:
        pred_detail_df = pd.DataFrame()
        pred_detail_save_path = None

    if metric_detail_rows:
        metric_detail_df = pd.DataFrame(metric_detail_rows)
        metric_detail_save_path = os.path.join(output_dir, "stage2_metric_detail_train_test.csv")
        metric_detail_df.to_csv(metric_detail_save_path, index=False, encoding="utf-8-sig")
        print(f"Training/testing error details saved to: {metric_detail_save_path}")

        
        metric_model_summary = (
            metric_detail_df.groupby(["split", "scheme_name", "model", "feature_num"], as_index=False)
            .agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                mse_mean=("mse", "mean"),
                mse_std=("mse", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                ae_mean=("ae_mean", "mean"),
                ae_std=("ae_mean", "std"),
                mape_mean=("mape", "mean"),
                mape_std=("mape", "std"),
            )
        )
        metric_model_summary_save_path = os.path.join(output_dir, "stage2_metric_summary_by_model_featurenum.csv")
        metric_model_summary.to_csv(metric_model_summary_save_path, index=False, encoding="utf-8-sig")
        print(f"Model-level summary errors saved to: {metric_model_summary_save_path}")
    else:
        metric_detail_df = pd.DataFrame()
        metric_detail_save_path = None

    
    
    
    best_dev_row = outer_eval_summary_df.iloc[0].to_dict()
    best_candidate_id = int(best_dev_row["candidate_id"])
    best_scheme_name = best_dev_row["scheme_name"]
    best_model_name = best_dev_row["model"]
    best_features = safe_json_loads(best_dev_row["features"])

    print("\n" + "=" * 100)
    print("Final candidate locked in the development stage")
    print("=" * 100)
    print(f"candidate_id: {best_candidate_id}")
    print(f"Scheme: {best_scheme_name}")
    print(f"Model: {best_model_name}")
    print(f"Feature count: {len(best_features)}")
    print(f"Feature subset: {best_features}")
    print(f"Outer R2: {best_dev_row['outer_r2_mean']:.4f} ± {best_dev_row['outer_r2_std']:.4f}")
    print(f"Outer RMSE: {best_dev_row['outer_rmse_mean']:.4f} ± {best_dev_row['outer_rmse_std']:.4f}")
    print(f"Outer MAE: {best_dev_row['outer_mae_mean']:.4f} ± {best_dev_row['outer_mae_std']:.4f}")

    
    
    
    print("\n" + "=" * 100)
    print("Stage 3: Retune on the full dataset and lock the final development-stage model")
    print("=" * 100)

    final_model_template, final_param_grid = tune_models[best_model_name]

    final_best_params, final_best_row, final_full_tuning_df = tune_model_with_groupkfold(
        X=X_all,
        y=y_all,
        groups=groups_all,
        features=best_features,
        model_name=best_model_name,
        model_template=final_model_template,
        param_grid=final_param_grid,
        n_splits=n_splits_inner
    )

    final_full_tuning_save_path = os.path.join(output_dir, "stage3_final_full_data_tuning_records.csv")
    final_full_tuning_df.to_csv(final_full_tuning_save_path, index=False, encoding="utf-8-sig")
    print(f"Final full-data tuning records saved to: {final_full_tuning_save_path}")

    final_locked_model = fit_model(
        X_train=X_all[best_features],
        y_train=y_all,
        model_template=final_model_template,
        params=final_best_params
    )

    model_save_path = os.path.join(output_dir, "final_locked_dev_model.joblib")
    joblib.dump(final_locked_model, model_save_path)
    print(f"Final locked model saved to: {model_save_path}")

    
    bundle = {
        "base_model": final_locked_model,
        "selected_features": best_features,
        "group_col": group_col,
        "target_col": target_col,
        "time_col": "Time",
        "selected_scheme_name": best_scheme_name,
        "selected_model_name": best_model_name,
        "selected_model_params": final_best_params
    }
    bundle_path = os.path.join(output_dir, "final_locked_dev_model_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"Final model bundle saved to: {bundle_path}")

    
    final_pred = final_locked_model.predict(X_all[best_features])
    final_metric = calc_metrics(y_all, final_pred)

    final_fit_df = df.copy()
    final_fit_df["pred"] = final_pred
    final_fit_df["residual"] = final_fit_df[target_col] - final_fit_df["pred"]
    final_fit_df["AE"] = np.abs(final_fit_df["residual"])
    denom = np.maximum(np.abs(final_fit_df[target_col].astype(float)), 1e-8)
    final_fit_df["APE"] = np.abs(final_fit_df["residual"] / denom) * 100.0

    final_fit_pred_save_path = os.path.join(output_dir, "stage3_final_model_fit_predictions.csv")
    final_fit_df.to_csv(final_fit_pred_save_path, index=False, encoding="utf-8-sig")
    print(f"Final model full-data fitted predictions saved to: {final_fit_pred_save_path}")

    final_fit_metric_df = pd.DataFrame([{
        "model": best_model_name,
        "feature_num": len(best_features),
        "features": safe_json_dumps(best_features),
        "r2": final_metric["r2"],
        "mse": final_metric["mse"],
        "rmse": final_metric["rmse"],
        "mae": final_metric["mae"],
        "ae_mean": final_metric["ae_mean"],
        "mape": final_metric["mape"]
    }])

    final_fit_metric_save_path = os.path.join(output_dir, "stage3_final_model_fit_metrics.csv")
    final_fit_metric_df.to_csv(final_fit_metric_save_path, index=False, encoding="utf-8-sig")
    print(f"Final model full-data fitted errors saved to: {final_fit_metric_save_path}")

    final_config = {
        "file_path": file_path,
        "output_dir": output_dir,
        "group_col": group_col,
        "target_col": target_col,
        "candidate_features_all": candidate_features,
        "analysis_schemes": analysis_schemes,
        "selected_scheme_name": best_scheme_name,
        "selected_model_name": best_model_name,
        "selected_features": best_features,
        "selected_feature_num": len(best_features),
        "selected_model_params": final_best_params,
        "dev_outer_r2_mean": float(best_dev_row["outer_r2_mean"]),
        "dev_outer_r2_std": float(best_dev_row["outer_r2_std"]),
        "dev_outer_rmse_mean": float(best_dev_row["outer_rmse_mean"]),
        "dev_outer_rmse_std": float(best_dev_row["outer_rmse_std"]),
        "dev_outer_mae_mean": float(best_dev_row["outer_mae_mean"]),
        "dev_outer_mae_std": float(best_dev_row["outer_mae_std"]),
        "full_data_cv_r2_mean": float(final_best_row["cv_r2_mean"]),
        "full_data_cv_r2_std": float(final_best_row["cv_r2_std"]),
        "full_data_cv_mse_mean": float(final_best_row["cv_mse_mean"]),
        "full_data_cv_mse_std": float(final_best_row["cv_mse_std"]),
        "full_data_cv_rmse_mean": float(final_best_row["cv_rmse_mean"]),
        "full_data_cv_rmse_std": float(final_best_row["cv_rmse_std"]),
        "full_data_cv_mae_mean": float(final_best_row["cv_mae_mean"]),
        "full_data_cv_mae_std": float(final_best_row["cv_mae_std"]),
        "full_data_cv_mape_mean": float(final_best_row["cv_mape_mean"]),
        "full_data_cv_mape_std": float(final_best_row["cv_mape_std"]),
        "full_data_fit_r2": float(final_metric["r2"]),
        "full_data_fit_mse": float(final_metric["mse"]),
        "full_data_fit_rmse": float(final_metric["rmse"]),
        "full_data_fit_mae": float(final_metric["mae"]),
        "full_data_fit_ae_mean": float(final_metric["ae_mean"]),
        "full_data_fit_mape": float(final_metric["mape"])
    }

    final_config_path = os.path.join(output_dir, "final_dev_model_config.json")
    with open(final_config_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(final_config), f, ensure_ascii=False, indent=2)

    final_summary_df = pd.DataFrame([{
        "selected_scheme_name": best_scheme_name,
        "selected_model_name": best_model_name,
        "selected_feature_num": len(best_features),
        "selected_features": safe_json_dumps(best_features),
        "selected_model_params": safe_json_dumps(final_best_params),
        "dev_outer_r2_mean": best_dev_row["outer_r2_mean"],
        "dev_outer_r2_std": best_dev_row["outer_r2_std"],
        "dev_outer_rmse_mean": best_dev_row["outer_rmse_mean"],
        "dev_outer_rmse_std": best_dev_row["outer_rmse_std"],
        "dev_outer_mae_mean": best_dev_row["outer_mae_mean"],
        "dev_outer_mae_std": best_dev_row["outer_mae_std"],
        "full_data_cv_r2_mean": final_best_row["cv_r2_mean"],
        "full_data_cv_r2_std": final_best_row["cv_r2_std"],
        "full_data_cv_mse_mean": final_best_row["cv_mse_mean"],
        "full_data_cv_mse_std": final_best_row["cv_mse_std"],
        "full_data_cv_rmse_mean": final_best_row["cv_rmse_mean"],
        "full_data_cv_rmse_std": final_best_row["cv_rmse_std"],
        "full_data_cv_mae_mean": final_best_row["cv_mae_mean"],
        "full_data_cv_mae_std": final_best_row["cv_mae_std"],
        "full_data_cv_mape_mean": final_best_row["cv_mape_mean"],
        "full_data_cv_mape_std": final_best_row["cv_mape_std"],
        "full_data_fit_r2": final_metric["r2"],
        "full_data_fit_mse": final_metric["mse"],
        "full_data_fit_rmse": final_metric["rmse"],
        "full_data_fit_mae": final_metric["mae"],
        "full_data_fit_ae_mean": final_metric["ae_mean"],
        "full_data_fit_mape": final_metric["mape"]
    }])

    final_summary_path = os.path.join(output_dir, "final_selected_dev_model_summary.csv")
    final_summary_df.to_csv(final_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("Final locked results (development stage)")
    print("=" * 100)
    print(f"Scheme: {best_scheme_name}")
    print(f"Model: {best_model_name}")
    print(f"Feature count: {len(best_features)}")
    print(f"Feature subset: {best_features}")
    print(f"Final parameters: {final_best_params}")
    print(f"Development-stage Outer R2: {best_dev_row['outer_r2_mean']:.4f} ± {best_dev_row['outer_r2_std']:.4f}")
    print(f"Development-stage Outer RMSE: {best_dev_row['outer_rmse_mean']:.4f} ± {best_dev_row['outer_rmse_std']:.4f}")
    print(f"Full-data GroupKFold CV R2: {final_best_row['cv_r2_mean']:.4f} ± {final_best_row['cv_r2_std']:.4f}")
    print(f"Full-data GroupKFold CV RMSE: {final_best_row['cv_rmse_mean']:.4f} ± {final_best_row['cv_rmse_std']:.4f}")

    print("\nKey output files:")
    print(f"1) Global best summary across three schemes: {scheme_global_best_save_path}")
    print(f"2) Three-scheme summary by feature count: {scheme_feature_count_summary_save_path}")
    print(f"3) Cross-scheme RMSE comparison plot: {scheme_compare_plot_path}")
    print(f"4) Stage 2 candidate pool: {candidate_pool_save_path}")
    print(f"5) Stage 2 outer summary: {outer_summary_save_path}")
    print(f"6) Stage 2 training/testing error details: {metric_detail_save_path if 'metric_detail_save_path' in locals() else 'Not generated'}")
    print(f"7) Stage 2 point-wise prediction details: {pred_detail_save_path if 'pred_detail_save_path' in locals() else 'Not generated'}")
    print(f"8) Stage 2 AE boxplot data: {ae_boxplot_path if 'ae_boxplot_path' in locals() else 'Not generated'}")
    print(f"9) Final model configuration: {final_config_path}")
    print(f"10) Final locked model: {model_save_path}")
    print(f"11) Final model bundle: {bundle_path}")
    print(f"12) Final summary table: {final_summary_path}")

else:
    print("\nStage 2 tuning was not run.")
    print("You have already obtained the full-range screening results for all three schemes and the RMSE vs. feature count plots.")
    print("After reviewing the plots and Stage 1 outputs, set RUN_STAGE2_TUNING to True to continue.")

print("\nAll processes completed.")