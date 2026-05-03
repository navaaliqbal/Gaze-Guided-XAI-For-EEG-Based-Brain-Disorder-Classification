import os
import json
import glob
import math
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Helpers: robust stats + effects
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))

def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def convex_hull_area(points):
    pts = [(float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    pts = sorted(set(pts))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i+1) % len(hull)]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0


# -----------------------------
# Feature extraction from fixations
# -----------------------------
def extract_features_from_fixations(fixations, roi_grid=(20, 10)):
    F = []
    for f in fixations:
        dur = safe_float(f.get("duration"))
        x = safe_float(f.get("x"))
        y = safe_float(f.get("y"))
        st = safe_float(f.get("start_time"))
        en = safe_float(f.get("end_time"))
        if np.isfinite(dur) and np.isfinite(x) and np.isfinite(y) and np.isfinite(st) and np.isfinite(en):
            F.append({"dur": dur, "x": x, "y": y, "st": st, "en": en})

    if len(F) < 2:
        return {
            "n_fix": len(F),
            "total_time": np.nan,
            "fps": np.nan,
            "fpm": np.nan,
            "fix_time_frac": np.nan,
            "dur_mean": np.nan,
            "dur_median": np.nan,
            "dur_std": np.nan,
            "dur_iqr": np.nan,
            "dur_cv": np.nan,
            "spread_bbox_area": np.nan,
            "spread_hull_area": np.nan,
            "spread_trace": np.nan,
            "spread_mean_dist_centroid": np.nan,
            "scanpath_total": np.nan,
            "scanpath_per_sec": np.nan,
            "step_mean": np.nan,
            "step_p90": np.nan,
            "tortuosity": np.nan,
            "backtrack_rate_x": np.nan,
            "roi_entropy": np.nan,
            "roi_occupancy": np.nan,
            "roi_top1": np.nan,
        }

    F = sorted(F, key=lambda z: z["st"])
    durs = np.array([z["dur"] for z in F], dtype=float)
    xs = np.array([z["x"] for z in F], dtype=float)
    ys = np.array([z["y"] for z in F], dtype=float)
    sts = np.array([z["st"] for z in F], dtype=float)
    ens = np.array([z["en"] for z in F], dtype=float)

    total_time = float(ens.max() - sts.min())
    if total_time <= 0:
        total_time = np.nan

    n_fix = len(F)
    fps = n_fix / total_time if np.isfinite(total_time) else np.nan
    fpm = 60.0 * fps if np.isfinite(fps) else np.nan
    fix_time = float(np.nansum(durs))
    fix_time_frac = fix_time / total_time if np.isfinite(total_time) and total_time > 0 else np.nan

    dur_mean = float(np.nanmean(durs))
    dur_median = float(np.nanmedian(durs))
    dur_std = float(np.nanstd(durs))
    dur_iqr = float(np.nanpercentile(durs, 75) - np.nanpercentile(durs, 25))
    dur_cv = (dur_std / dur_mean) if dur_mean > 0 else np.nan

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    spread_bbox_area = (x_max - x_min) * (y_max - y_min)

    points = list(zip(xs, ys))
    spread_hull_area = convex_hull_area(points)

    var_x = float(np.nanvar(xs))
    var_y = float(np.nanvar(ys))
    spread_trace = var_x + var_y

    cx, cy = float(np.nanmean(xs)), float(np.nanmean(ys))
    d_cent = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    spread_mean_dist_centroid = float(np.nanmean(d_cent))

    dx = np.diff(xs)
    dy = np.diff(ys)
    steps = np.sqrt(dx * dx + dy * dy)
    scanpath_total = float(np.nansum(steps))
    scanpath_per_sec = scanpath_total / total_time if np.isfinite(total_time) and total_time > 0 else np.nan
    step_mean = float(np.nanmean(steps))
    step_p90 = float(np.nanpercentile(steps, 90))

    direct = math.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
    tortuosity = (scanpath_total / direct) if direct > 0 else np.nan
    backtrack_rate_x = float(np.mean(np.diff(xs) < 0))

    nx, ny = roi_grid
    if x_max - x_min < 1e-9 or y_max - y_min < 1e-9:
        roi_entropy, roi_occupancy, roi_top1 = np.nan, np.nan, np.nan
    else:
        ix = np.clip(((xs - x_min) / (x_max - x_min) * nx).astype(int), 0, nx - 1)
        iy = np.clip(((ys - y_min) / (y_max - y_min) * ny).astype(int), 0, ny - 1)
        bins = ix + nx * iy
        counts = np.bincount(bins, minlength=nx * ny)
        roi_entropy = float(shannon_entropy(counts))
        roi_occupancy = float(np.sum(counts > 0) / (nx * ny))
        roi_top1 = float(np.max(counts) / np.sum(counts)) if counts.sum() > 0 else np.nan

    return {
        "n_fix": n_fix,
        "total_time": total_time,
        "fps": fps,
        "fpm": fpm,
        "fix_time_frac": fix_time_frac,
        "dur_mean": dur_mean,
        "dur_median": dur_median,
        "dur_std": dur_std,
        "dur_iqr": dur_iqr,
        "dur_cv": dur_cv,
        "spread_bbox_area": spread_bbox_area,
        "spread_hull_area": spread_hull_area,
        "spread_trace": spread_trace,
        "spread_mean_dist_centroid": spread_mean_dist_centroid,
        "scanpath_total": scanpath_total,
        "scanpath_per_sec": scanpath_per_sec,
        "step_mean": step_mean,
        "step_p90": step_p90,
        "tortuosity": tortuosity,
        "backtrack_rate_x": backtrack_rate_x,
        "roi_entropy": roi_entropy,
        "roi_occupancy": roi_occupancy,
        "roi_top1": roi_top1,
    }


def load_fixations_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("fixations", [])

# -----------------------------
# Build dataset: exact filename matching
# -----------------------------
def extract_subject_id(filename):
    """Extract subject ID from filename like '0000002_P0_S0.json' -> '0000002'"""
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    return parts[0] if parts else filename

def build_feature_table(fix_folder, label_csv, json_ext="*.json", roi_grid=(20, 10)):
    # Read CSV (expect columns 'File' and 'Label')
    labels = pd.read_csv(label_csv, dtype={"File": str})
    labels["File"] = labels["File"].str.strip()
    labels["Label"] = labels["Label"].astype(str).str.strip().str.lower()

    label_map = dict(zip(labels["File"], labels["Label"]))
    print(f"[INFO] Loaded {len(label_map)} labels from CSV")
    print(f"[INFO] First 5 labels: {list(label_map.items())[:5]}")

    rows = []
    json_paths = glob.glob(os.path.join(fix_folder, json_ext))
    print(f"[INFO] Found {len(json_paths)} JSON files in {fix_folder}")
    if json_paths:
        print(f"[INFO] Example JSON: {os.path.basename(json_paths[0])}")

    matched = 0
    for p in json_paths:
        base = os.path.basename(p)          # e.g., "0000002_P0_S0.json"
        if base not in label_map:
            continue
        matched += 1
        fix = load_fixations_from_file(p)
        feats = extract_features_from_fixations(fix, roi_grid=roi_grid)
        feats["file_id"] = base
        feats["label"] = label_map[base]
        feats["json_file"] = base
        feats["subject_id"] = extract_subject_id(base)
        rows.append(feats)

    print(f"[INFO] Matched {matched} JSON files to labels")
    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Stats comparison + simple model
# -----------------------------
def compare_groups(df, label_col="label", pos_label="abnormal"):
    df = df.copy()
    df["y"] = (df[label_col] == pos_label).astype(int)

    feature_cols = [c for c in df.columns if c not in ["file_id", "json_file", "label", "y", "subject_id"]]
    results = []

    for c in feature_cols:
        x_abn = df.loc[df["y"] == 1, c].dropna().values
        x_norm = df.loc[df["y"] == 0, c].dropna().values
        if len(x_abn) < 5 or len(x_norm) < 5:
            continue

        try:
            _, p = mannwhitneyu(x_abn, x_norm, alternative="two-sided")
        except Exception:
            p = np.nan

        delta = cliffs_delta(x_abn, x_norm)

        results.append({
            "feature": c,
            "mean_abnormal": float(np.mean(x_abn)),
            "mean_normal": float(np.mean(x_norm)),
            "median_abnormal": float(np.median(x_abn)),
            "median_normal": float(np.median(x_norm)),
            "p_mwu": p,
            "cliffs_delta": delta
        })

    res = pd.DataFrame(results).sort_values("p_mwu", ascending=True)
    return res


def run_simple_auc(df, label_col="label", pos_label="abnormal", group_col=None):
    df = df.copy()
    df["y"] = (df[label_col] == pos_label).astype(int)

    feature_cols = [c for c in df.columns if c not in ["file_id", "json_file", "label", "y", "subject_id"]]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["y"].values

    keep = X.columns[X.isna().mean() < 0.3]
    X = X[keep].copy()
    X = X.fillna(X.median(numeric_only=True))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    if group_col is not None and group_col in df.columns:
        groups = df[group_col].values
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="roc_auc")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    return float(np.mean(scores)), float(np.std(scores)), list(keep)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    FIX_FOLDER = r"C:\Users\S.S.T\Documents\VsCode\Gaze Data Collection\64\fixations_83_0.35_70_new_segments"
    LABEL_CSV  = r"C:\Users\S.S.T\Documents\VsCode\Gaze Data Collection\64\eeglabels.csv"
    ROI_GRID = (20, 10)

    df = build_feature_table(FIX_FOLDER, LABEL_CSV, json_ext="*.json", roi_grid=ROI_GRID)

    if len(df) == 0:
        print("❌ No samples matched. Check that:")
        print("   - The folder path is correct")
        print("   - JSON filenames exactly match the 'File' column in CSV (including .json)")
        print("   - CSV column names are 'File' and 'Label' (case-sensitive)")
        exit()

    print("\n✅ Loaded samples:", len(df))
    print(df["label"].value_counts(dropna=False))

    # Save features
    df.to_csv("gaze_fixation_features.csv", index=False)
    print("Saved: gaze_fixation_features.csv")

    # Group comparison
    stats_df = compare_groups(df)
    stats_df.to_csv("feature_group_stats.csv", index=False)
    print("Saved: feature_group_stats.csv")

    print("\nTop features by MWU p-value:")
    print(stats_df.head(15).to_string(index=False))

    # AUC with subject-level grouping
    mean_auc, std_auc, used_features = run_simple_auc(df, group_col="subject_id")
    print(f"\n5-fold GroupKFold AUC (logistic regression): {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Used {len(used_features)} features: {used_features[:10]}...")