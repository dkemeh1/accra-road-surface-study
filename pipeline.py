"""
Accra Road Surface Mapping Pipeline

This script reproduces the machine learning workflow used in the study:

'Mapping Road Surface Conditions in Accra Using Multi-Sensor Satellite Data'

Pipeline Steps:
1. Extract OSM road surface tags
2. Create weak training labels
3. Segment road network into 100 m segments
4. Extract satellite features
5. Train models using spatial cross-validation
6. Predict road surface classes
7. Detect road surface changes
8. Compare model experiments

Author: Desmond Kemeh
Institution: Ariel University
"""

import os
import math
import time
import glob
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import joblib
import xlsxwriter

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from openpyxl import load_workbook
from openpyxl.styles import Font
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# 0) GLOBAL CONFIG
# ============================================================
BASE_DIR = "./data"
IMAGERY_DIR = os.path.join(BASE_DIR, "imagery")

YEAR_A = "2018"
YEAR_B = "2024"  # change to "2025" when you have it

SNAPSHOT_DATE_A = f"{YEAR_A}-11-30"
SNAPSHOT_DATE_B = f"{YEAR_B}-12-31"

SNAP_A = os.path.join(BASE_DIR, f"snapshot_{YEAR_A}_{YEAR_A}-12-31.geojson")
SNAP_B = os.path.join(BASE_DIR, f"snapshot_{YEAR_B}_{YEAR_B}-12-31.geojson")

WINDOW_DAYS = 30


# CRS + segmentation
METRIC_CRS = "EPSG:32630"   # UTM 30N
SEG_LEN_M = 100.0

# Step 4 batching
BATCH_SIZE = 6000
SLEEP_BETWEEN = 1
MAX_RETRIES = 5

# Step 5 spatial split (tile grouping)
TILE_SIZE_M = 2000

# Step 7 change detection
MAX_MATCH_DIST_M = 30

# ============================================================
# Step 5/6 KNOBS
# ============================================================
FEATURE_CLIP_QLOW = 0.01
FEATURE_CLIP_QHIGH = 0.99
THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

# CONSISTENT RULE ACROSS ALL YEARS/EXPERIMENTS (DO NOT CHANGE AFTER LOOKING):
# - Step 5 reports BOTH best-F1 and best-BALACC
# - Step 6 uses ONLY this selection
THRESH_SELECTION_FOR_PREDICTION = "BALACC"   # "BALACC" or "F1"

# Guardrail against class-collapse thresholds
MIN_POS_RATE = 0.05   # at least 5% predicted positives
MAX_POS_RATE = 0.95   # at most 95% predicted positives

MODEL_TYPES = ["RF", "XGB", "LGBM"]   # choose any subset

RF_PARAMS = dict(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_features=0.5,
    min_samples_leaf=1,
    min_samples_split=4
)

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

# ============================================================
# 0B) WHAT TO RUN
# ============================================================
RUN_EXPERIMENTS = [ "L8","S1","S2"]

# Turn steps ON (True) or OFF(False) depending on what you want to run
RUN_STEP_1 = True
RUN_STEP_2 = True
RUN_STEP_3 = True
RUN_STEP_4 = True
RUN_STEP_5_6 = True
RUN_STEP_7 = True
RUN_STEP_8 = True


# ============================================================
# 1) COMMON HELPERS
# ============================================================
def ensure_crs_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def get_surface_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    direct = gdf["surface"] if "surface" in gdf.columns else pd.Series([None]*len(gdf), index=gdf.index)
    if "tags" in gdf.columns:
        nested = gdf["tags"].apply(lambda x: x.get("surface") if isinstance(x, dict) else None)
    else:
        nested = pd.Series([None]*len(gdf), index=gdf.index)
    return direct.where(direct.notna(), nested)

def get_highway_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    direct = gdf["highway"] if "highway" in gdf.columns else pd.Series([None]*len(gdf), index=gdf.index)
    if "tags" in gdf.columns:
        nested = gdf["tags"].apply(lambda x: x.get("highway") if isinstance(x, dict) else None)
    else:
        nested = pd.Series([None]*len(gdf), index=gdf.index)
    return direct.where(direct.notna(), nested)

def extract_osm_id(gdf: gpd.GeoDataFrame) -> pd.Series:
    for c in ["@id", "id", "osm_id", "osmid", "osmId", "@osmId"]:
        if c in gdf.columns:
            return gdf[c]
    if "tags" in gdf.columns:
        return gdf["tags"].apply(lambda x: x.get("@id") if isinstance(x, dict) else None)
    return pd.Series([None]*len(gdf), index=gdf.index)

def to_lines(geom):
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
    return None

def split_line_into_segments(line: LineString, seg_len: float):
    length = line.length
    if length <= 0:
        return []
    n = max(1, int(math.ceil(length / seg_len)))
    segs = []
    for i in range(n):
        start_d = i * seg_len
        end_d = min((i + 1) * seg_len, length)
        if end_d - start_d < 1.0:
            continue
        steps = max(2, int((end_d - start_d) / 5))
        pts = [line.interpolate(start_d + (end_d - start_d) * (k / steps)) for k in range(steps + 1)]
        segs.append(LineString(pts))
    return segs


# ============================================================
# 2) EXPERIMENT PATHS
# ============================================================
def make_experiment_dirs(experiment_name: str) -> dict:
    EXP_ROOT = os.path.join(BASE_DIR, "experiments", experiment_name)
    OUT = {
        "EXP_ROOT": EXP_ROOT,
        "OUT_STEP1": os.path.join(EXP_ROOT, "surface_tags_only"),
        "OUT_STEP2": os.path.join(EXP_ROOT, "step2_labels"),
        "OUT_STEP3": os.path.join(EXP_ROOT, "step3_segments"),
        "OUT_STEP4_A": os.path.join(EXP_ROOT, f"step4_features_{YEAR_A}_6000"),
        "OUT_STEP4_B": os.path.join(EXP_ROOT, f"step4_features_{YEAR_B}_6000"),
        "OUT_STEP4_MERGED": os.path.join(EXP_ROOT, "step4_features"),
        "OUT_STEP5": os.path.join(EXP_ROOT, "step5_models"),
        "OUT_STEP6": os.path.join(EXP_ROOT, "step6_predictions"),
        "OUT_STEP7": os.path.join(EXP_ROOT, "step7_change_detection"),
        "OUT_STEP8": os.path.join(EXP_ROOT, "step8_compare"),
    }
    for d in OUT.values():
        os.makedirs(d, exist_ok=True)
    return OUT


# ============================================================
# STEP 1 — Unique surface tags
# ============================================================
def step1_extract_unique_surfaces(snapshot_path: str, year_label: str, OUT_STEP1: str):
    print("\n" + "="*70)
    print(f"STEP 1 — UNIQUE SURFACE TAG VALUES: {year_label}")
    print("="*70)

    gdf = ensure_crs_wgs84(gpd.read_file(snapshot_path))
    surface = get_surface_series(gdf)
    unique_vals = (
        surface.dropna().astype(str).str.strip()
        .replace("", pd.NA).dropna().unique().tolist()
    )
    unique_vals = sorted(set(unique_vals), key=lambda s: s.lower())

    out_txt = os.path.join(OUT_STEP1, f"surface_tags_{year_label}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for v in unique_vals:
            f.write(v + "\n")
    print(f"Saved: {out_txt}")


# ============================================================
# STEP 2 — Label training data (OSM weak supervision)
# ============================================================
PAVED_SET = {"asphalt", "concrete", "paved", "paving_stones", "sett", "concrete:plates"}
UNPAVED_SET = {"unpaved", "pebblestone", "grass", "metal", "wood", "rock",
               "gravel", "dirt", "earth", "ground", "sand", "mud", "fine_gravel", "compacted"}

def step2_label_snapshot(in_path: str, year_label: str, OUT_STEP2: str):
    print("\n" + "="*70)
    print(f"STEP 2 — TRAINING LABELS (PAVED=1 / UNPAVED=0): {year_label}")
    print("="*70)

    gdf = ensure_crs_wgs84(gpd.read_file(in_path))
    gdf["highway_extracted"] = get_highway_series(gdf)
    roads = gdf[gdf["highway_extracted"].notna()].copy()

    roads["surface_extracted"] = get_surface_series(roads).astype(str).str.strip().str.lower()
    roads.loc[roads["surface_extracted"].isin(["none", "nan", ""]), "surface_extracted"] = pd.NA
    roads = roads[roads["surface_extracted"].notna()].copy()

    roads["label"] = pd.NA
    roads.loc[roads["surface_extracted"].isin(PAVED_SET), "label"] = 1
    roads.loc[roads["surface_extracted"].isin(UNPAVED_SET), "label"] = 0

    train = roads[roads["label"].notna()].copy()
    train["label"] = train["label"].astype(int)
    train["year"] = int(year_label)

    out_gpkg = os.path.join(OUT_STEP2, f"train_{year_label}.gpkg")
    train.to_file(out_gpkg, layer="train", driver="GPKG")
    print(f"Saved: {out_gpkg} | rows={len(train):,}")


# ============================================================
# STEP 3 — Segment all roads + make train segments
# ============================================================
def step3_segment_all_roads(snapshot_path: str, year_label: str, OUT_STEP3: str):
    print("\n" + "="*70)
    print(f"STEP 3A — SEGMENT ALL ROADS (every {SEG_LEN_M}m): {year_label}")
    print("="*70)

    gdf = ensure_crs_wgs84(gpd.read_file(snapshot_path))
    gdf["highway"] = get_highway_series(gdf)
    gdf["surface"] = get_surface_series(gdf)
    gdf["osm_id"] = extract_osm_id(gdf)

    roads = gdf[gdf["highway"].notna()].copy()
    roads = roads[~roads.geometry.isna()].copy()
    roads = roads.to_crs(METRIC_CRS)
    roads["geometry"] = roads["geometry"].apply(to_lines)
    roads = roads[roads["geometry"].notna()].copy()

    rows = []
    for idx, r in roads.iterrows():
        segs = split_line_into_segments(r.geometry, SEG_LEN_M)
        for j, seg in enumerate(segs):
            rows.append({
                "seg_id": f"{year_label}_{idx}_{j}",
                "year": int(year_label),
                "osm_id": r.get("osm_id"),
                "highway": str(r.get("highway")),
                "surface": r.get("surface"),
                "geometry": seg
            })

    seg_gdf = gpd.GeoDataFrame(rows, crs=METRIC_CRS)
    out_all = os.path.join(OUT_STEP3, f"segments_{year_label}_all.gpkg")
    seg_gdf.to_file(out_all, layer="segments_all", driver="GPKG")
    print(f"Saved ALL segments: {len(seg_gdf):,} -> {out_all}")
    return out_all

def step3_make_train_segments(seg_all_gpkg: str, train_gpkg: str, year_label: str, OUT_STEP3: str):
    print("\n" + "="*70)
    print(f"STEP 3B — BUILD TRAIN SEGMENTS (intersects labeled roads): {year_label}")
    print("="*70)

    seg_all = gpd.read_file(seg_all_gpkg, layer="segments_all").to_crs(METRIC_CRS)
    train = ensure_crs_wgs84(gpd.read_file(train_gpkg, layer="train")).to_crs(METRIC_CRS)

    train_small = train[["label", "geometry"]].copy()
    joined = gpd.sjoin(seg_all, train_small, how="inner", predicate="intersects") \
                .drop(columns=["index_right"], errors="ignore")

    out_train = os.path.join(OUT_STEP3, f"segments_{year_label}_train.gpkg")
    joined.to_file(out_train, layer="segments_train", driver="GPKG")
    print(f"Saved TRAIN segments: {len(joined):,} -> {out_train}")
    return out_train



def make_model(model_type: str):
    model_type = model_type.upper()

    if model_type == "RF":
        return RandomForestClassifier(**RF_PARAMS)

    elif model_type == "XGB":
        return XGBClassifier(**XGB_PARAMS)

    elif model_type == "LGBM":
        return LGBMClassifier(**LGBM_PARAMS)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# NEW STEP 4 — LOCAL RASTER FEATURE EXTRACTION
# ============================================================

def build_optical_from_export(raster_path):

    with rasterio.open(raster_path) as src:
        stack = src.read().astype("float32")
        profile = src.profile

    BLUE  = stack[0]
    GREEN = stack[1]
    RED   = stack[2]
    NIR   = stack[3]
    SWIR1 = stack[4]
    SWIR2 = stack[5]

    if stack.shape[0] >= 8:
        NDVI = stack[6]
        NDBI = stack[7]
    else:
        NDVI = (NIR - RED) / (NIR + RED + 1e-6)
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR + 1e-6)

    BRIGHT = (RED + NIR + SWIR1) / 3.0

    final_stack = np.stack([
        BLUE, GREEN, RED, NIR,
        SWIR1, SWIR2,
        NDVI, NDBI, BRIGHT
    ])

    feature_names = [
        "BLUE","GREEN","RED","NIR",
        "SWIR1","SWIR2",
        "NDVI","NDBI","BRIGHT"
    ]

    return final_stack, profile, feature_names


def build_s1_from_export(raster_path):

    with rasterio.open(raster_path) as src:
        stack = src.read().astype("float32")
        profile = src.profile

    VV = stack[0]
    VH = stack[1]

    if np.nanmean(VV) < 0:
        VV_db = VV
        VH_db = VH
    else:
        VV_db = 10 * np.log10(VV + 1e-6)
        VH_db = 10 * np.log10(VH + 1e-6)

    VVminusVH = VV_db - VH_db
    VVdivVH   = VV_db / (VH_db + 1e-6)

    final_stack = np.stack([
        VV_db,
        VH_db,
        VVminusVH,
        VVdivVH
    ])

    feature_names = [
        "VV_db","VH_db","VVminusVH","VVdivVH"
    ]

    return final_stack, profile, feature_names

def step4_extract_local(seg_all_gpkg, raster_path, sensor, out_csv):

    # --------------------------------------------------
    # 1) Load segments
    # --------------------------------------------------
    seg = gpd.read_file(seg_all_gpkg, layer="segments_all")

    # --------------------------------------------------
    # 2) FORCE projection to METRIC CRS before centroid
    # --------------------------------------------------
    if seg.crs != METRIC_CRS:
        seg = seg.to_crs(METRIC_CRS)

    # --------------------------------------------------
    # 3) Open raster
    # --------------------------------------------------
    with rasterio.open(raster_path) as src:

        # If raster not metric, reproject segments to raster CRS AFTER centroid-safe projection
        if src.crs != METRIC_CRS:
            seg = seg.to_crs(src.crs)

        # --------------------------------------------------
        # 4) Build feature stack
        # --------------------------------------------------
        if sensor in ["L8", "S2"]:
            stack, profile, feature_names = build_optical_from_export(raster_path)
        elif sensor == "S1":
            stack, profile, feature_names = build_s1_from_export(raster_path)
        else:
            raise ValueError("Unknown sensor")

        # --------------------------------------------------
        # 5) Compute centroids (NOW SAFE)
        # --------------------------------------------------
        centroids = seg.geometry.centroid
        coords = [(pt.x, pt.y) for pt in centroids]

        rows = []

        for x, y in coords:
            row, col = src.index(x, y)

            if 0 <= row < stack.shape[1] and 0 <= col < stack.shape[2]:
                pixel_vals = stack[:, row, col]
            else:
                pixel_vals = [np.nan] * stack.shape[0]

            rows.append(pixel_vals)

    # --------------------------------------------------
    # 6) Save CSV
    # --------------------------------------------------
    df = pd.DataFrame(rows, columns=feature_names)

    df["seg_id"] = seg["seg_id"].values
    df["year"] = seg["year"].values
    df["highway"] = seg["highway"].values
    df["surface"] = seg["surface"].values

    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

# ============================================================
# STEP 5 — Train + VALIDATE (Spatial CV) [PUBLISH-SAFE: NO LEAKAGE]
# ============================================================
def add_tile_group(gdf):
    cent = gdf.geometry.centroid
    gdf["tx"] = (cent.x // TILE_SIZE_M).astype(int)
    gdf["ty"] = (cent.y // TILE_SIZE_M).astype(int)
    gdf["tile_id"] = gdf["tx"].astype(str) + "_" + gdf["ty"].astype(str)
    return gdf

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    keep_non_num = {"seg_id", "year", "highway", "surface"}
    feat_cols = [c for c in df.columns if c not in keep_non_num and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols

def _compute_clip_bounds(X: pd.DataFrame, qlow=FEATURE_CLIP_QLOW, qhigh=FEATURE_CLIP_QHIGH) -> dict:
    bounds = {}
    for c in X.columns:
        s = X[c].dropna()
        if len(s) == 0:
            continue
        lo = float(s.quantile(qlow))
        hi = float(s.quantile(qhigh))
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds

def _clip_with_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    Xc = X.copy()
    for c, (lo, hi) in bounds.items():
        if c in Xc.columns:
            Xc[c] = Xc[c].clip(lower=lo, upper=hi)
    return Xc

def _passes_guardrail(pred: np.ndarray) -> bool:
    pos_rate = float(np.mean(pred == 1))
    return (pos_rate >= MIN_POS_RATE) and (pos_rate <= MAX_POS_RATE)

def _mcc_from_cm(cm: np.ndarray) -> float:
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    fn, tp = float(cm[1, 0]), float(cm[1, 1])
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom

def _cm_row_normalized(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums

def _metrics_from_pred(y_true: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = float(accuracy_score(y_true, pred))
    balacc = float(balanced_accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))  # TPR
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(_mcc_from_cm(cm))
    pos_rate = float(np.mean(pred == 1))
    cm_norm = _cm_row_normalized(cm)
    return {
        "acc": acc, "balacc": balacc, "precision": prec, "recall": rec,
        "tpr": rec, "tnr": tnr, "mcc": mcc, "pos_rate": pos_rate,
        "cm": cm, "cm_norm": cm_norm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def _search_thresholds(y_true: np.ndarray, proba: np.ndarray, grid=THRESH_GRID):

    best_score = -1.0
    best_row = None

    for t in grid:
        pred = (proba >= t).astype(int)

        if not _passes_guardrail(pred):
            continue

        m = _metrics_from_pred(y_true, pred)

        if m["balacc"] > best_score:
            best_score = m["balacc"]
            best_row = {
                "t": float(t),
                **m
            }

    # If guardrail removed everything → fallback 0.50
    if best_row is None:
        t = 0.50
        pred = (proba >= t).astype(int)
        m = _metrics_from_pred(y_true, pred)
        best_row = {
            "t": float(t),
            **m
        }

    return best_row

def _print_threshold_report(year_label: str, tag: str, row: dict):

    cm = row["cm"]
    cm_norm = row["cm_norm"]

    print("\n" + "-"*60)
    print(f"[{year_label}] BEST BALACC threshold = {row['t']:.2f}")
    print(f"Balanced Accuracy = {row['balacc']:.3f}")
    print(f"MCC = {row['mcc']:.3f}")

    print("\nConfusion Matrix (RAW):")
    print(cm)

    print("\nConfusion Matrix (% by TRUE class):")
    print(np.round(cm_norm * 100, 2))

    print("-"*60)


def step5_train_model_with_validation(year_label, train_seg_gpkg, feat_csv, model_out, eval_out_csv, model_type):
    print("\n" + "="*70)
    print(f"STEP 5 — TRAIN + VALIDATE (Spatial CV) [PUBLISH-SAFE v3.1]: {year_label}")
    print("="*70)

    seg_train = gpd.read_file(train_seg_gpkg, layer="segments_train").to_crs(METRIC_CRS)
    seg_train = add_tile_group(seg_train)

    Xdf, feat_cols = load_features(feat_csv)
    df = seg_train[["seg_id", "label", "tile_id"]].merge(Xdf, on="seg_id", how="inner").copy()

    if df.empty:
        raise RuntimeError(f"[{year_label}] No training rows after merge. Check seg_id match between GPKG and CSV.")

    y = df["label"].astype(int).values
    groups = df["tile_id"].astype(str).values

    Xraw = df[feat_cols].copy()

    gkf = GroupKFold(n_splits=5)
    oof_proba = np.full(shape=(len(Xraw),), fill_value=np.nan, dtype=float)

    fold_rows = []
    for fold, (tr, te) in enumerate(gkf.split(Xraw, y, groups), start=1):
        # --- Fold-safe preprocessing (NO LEAKAGE) ---
        X_tr = Xraw.iloc[tr].copy()
        X_te = Xraw.iloc[te].copy()

        med_tr = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med_tr)
        X_te = X_te.fillna(med_tr)

        clip_bounds_tr = _compute_clip_bounds(X_tr, FEATURE_CLIP_QLOW, FEATURE_CLIP_QHIGH)
        X_tr = _clip_with_bounds(X_tr, clip_bounds_tr)
        X_te = _clip_with_bounds(X_te, clip_bounds_tr)

        clf = make_model(model_type)
        clf.fit(X_tr, y[tr])

        proba = clf.predict_proba(X_te)[:, 1]
        oof_proba[te] = proba

        pred50 = (proba >= 0.50).astype(int)
        m50 = _metrics_from_pred(y[te], pred50)

        fold_rows.append({
            "year": int(year_label),
            "fold": fold,
            "balacc@0.50": float(m50["balacc"]),
            "mcc@0.50": float(m50["mcc"]),
        })

        print(f"Fold {fold}: balacc@0.50={m50['balacc']:.3f}  mcc@0.50={m50['mcc']:.3f}")


    # --- OOF threshold tuning (guardrail-safe) ---
    mask = ~np.isnan(oof_proba)
    y_oof = y[mask]
    p_oof = oof_proba[mask]

    row_ba = _search_thresholds(y_oof, p_oof, THRESH_GRID)
    _print_threshold_report(year_label, "BALACC", row_ba)


    # Means across folds at fixed 0.50
    mean_balacc_050 = float(np.mean([r["balacc@0.50"] for r in fold_rows]))
    mean_mcc_050 = float(np.mean([r["mcc@0.50"] for r in fold_rows]))

    eval_df = pd.DataFrame(fold_rows)

    summary_mean = pd.DataFrame([{
        "year": int(year_label),
        "mean_balacc@0.50": mean_balacc_050,
        "mean_mcc@0.50": mean_mcc_050,
        "n_train_rows": int(len(df)),
        "n_features": int(len(feat_cols)),
    }])

    def _summary(tag, row):
        cm = row["cm"]
        fb = bool(row.get("guardrail_fallback", False))
        return {
            "year": int(year_label),
            "fold": f"OOF_BEST_{tag}",
            "best_threshold": float(row["t"]),
            "accuracy@bestT": float(row["acc"]),
            "balacc@bestT": float(row["balacc"]),
            "precision@bestT": float(row["precision"]),
            "recall_tpr@bestT": float(row["tpr"]),
            "tnr@bestT": float(row["tnr"]),
            "mcc@bestT": float(row["mcc"]),
            "pos_rate@bestT": float(row["pos_rate"]),
            "cm_tn": int(cm[0, 0]),
            "cm_fp": int(cm[0, 1]),
            "cm_fn": int(cm[1, 0]),
            "cm_tp": int(cm[1, 1]),
            "n_train_rows": int(len(df)),
            "n_features": int(len(feat_cols)),
            "guardrail_fallback": fb,
            "note": "OOF tuned threshold (from OOF predictions; not an external holdout). If guardrail_fallback=True, threshold forced to 0.50."
        }

    cm_norm = row_ba["cm_norm"]

    summary = pd.DataFrame([{
        "year": int(year_label),
        "best_threshold": float(row_ba["t"]),
        "balanced_accuracy": float(row_ba["balacc"]),
        "mcc": float(row_ba["mcc"]),

        # RAW
        "tn": int(row_ba["tn"]),
        "fp": int(row_ba["fp"]),
        "fn": int(row_ba["fn"]),
        "tp": int(row_ba["tp"]),

        # PERCENTAGES (row-normalized)
        "tn_pct": round(float(cm_norm[0, 0] * 100), 2),
        "fp_pct": round(float(cm_norm[0, 1] * 100), 2),
        "fn_pct": round(float(cm_norm[1, 0] * 100), 2),
        "tp_pct": round(float(cm_norm[1, 1] * 100), 2),
    }])

    summary.to_csv(eval_out_csv, index=False)
    print(f"Saved simplified eval CSV: {eval_out_csv}")

    # --- Train FINAL model on ALL training data (consistent stats saved for Step 6) ---
    med_full = Xraw.median(numeric_only=True)
    X_full = Xraw.fillna(med_full)
    clip_bounds_full = _compute_clip_bounds(X_full, FEATURE_CLIP_QLOW, FEATURE_CLIP_QHIGH)
    X_full = _clip_with_bounds(X_full, clip_bounds_full)

    final = make_model(model_type)
    final.fit(X_full, y)

    chosen_t = float(row_ba["t"])
    chosen_tag = "BALACC"

    joblib.dump({
        "model": final,
        "features": feat_cols,
        "train_medians": med_full.to_dict(),
        "clip_bounds": clip_bounds_full,
        "threshold_used": float(chosen_t),
        "threshold_mode": chosen_tag,
        "provenance": "Publish-safe v3.1: fold-safe CV preprocessing; BALACC-only threshold selection"
    }, model_out)

    print(f"Saved model bundle: {model_out}  (Step 6 will use {chosen_tag} threshold = {chosen_t:.2f})")


# ============================================================
# STEP 6 — Predict ALL + QC (NOT accuracy) [CONSISTENT WITH TRAINING]
# ============================================================
def _ensure_feature_columns(df: pd.DataFrame, feat_cols: list, fill_values: dict) -> pd.DataFrame:
    out = df.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = np.nan
        if c in fill_values:
            out[c] = out[c].fillna(fill_values[c])
    return out

def step6_predict_all_with_qc(year_label, seg_all_gpkg, feat_csv, model_path, out_gpkg, qc_out_csv):
    print("\n" + "="*70)
    print(f"STEP 6 — PREDICT ALL + QC [PUBLISH-SAFE v3.1]: {year_label}")
    print("="*70)

    seg_all = gpd.read_file(seg_all_gpkg, layer="segments_all").to_crs(METRIC_CRS)
    Xdf, _ = load_features(feat_csv)

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    feat_cols = bundle["features"]
    clip_bounds = bundle.get("clip_bounds", {})
    train_medians = bundle.get("train_medians", {})

    threshold = float(bundle.get("threshold_used", 0.50))
    threshold_mode = str(bundle.get("threshold_mode", "0.50"))

    df = seg_all.merge(Xdf, on="seg_id", how="left")

    miss_feat = float(df[feat_cols].isna().mean().mean()) if all(c in df.columns for c in feat_cols) else np.nan

    df = _ensure_feature_columns(df, feat_cols, train_medians)
    Xmat = df[feat_cols].copy()
    Xmat = Xmat.fillna(0.0)

    if isinstance(clip_bounds, dict) and clip_bounds:
        Xmat = _clip_with_bounds(Xmat, clip_bounds)

    proba = clf.predict_proba(Xmat)[:, 1]
    pred = (proba >= threshold).astype(int)

    df["p_paved"] = proba
    df["pred_label"] = pred
    df["pred_surface"] = df["pred_label"].map({1: "paved", 0: "unpaved"})
    df["threshold_used"] = threshold
    df["threshold_mode"] = threshold_mode

    out = gpd.GeoDataFrame(df, crs=METRIC_CRS)
    out.to_file(out_gpkg, layer="predicted", driver="GPKG")
    print(f"Saved predictions: {out_gpkg}")

    qc = {
        "year": int(year_label),
        "threshold_used": float(threshold),
        "threshold_mode": threshold_mode,
        "n_segments": int(len(df)),
        "mean_missing_feature_rate": float(miss_feat) if pd.notna(miss_feat) else np.nan,
        "p_paved_mean": float(np.mean(proba)),
        "p_paved_std": float(np.std(proba)),
        "p_paved_min": float(np.min(proba)),
        "p_paved_max": float(np.max(proba)),
        "pct_pred_paved": float(np.mean(pred == 1) * 100.0),
        "pct_pred_unpaved": float(np.mean(pred == 0) * 100.0),
    }
    pd.DataFrame([qc]).to_csv(qc_out_csv, index=False)
    print(f"Saved QC CSV: {qc_out_csv}")


# ============================================================
# STEP 7 — Change detection (within SAME experiment)
# ============================================================
LAYER_BY_CLASS = {
    "Upgrade_unpaved_to_paved": "upgrade_unpaved_to_paved",
    "Downgrade_paved_to_unpaved": "downgrade_paved_to_unpaved",
    "Stable_paved": "stable_paved",
    "Stable_unpaved": "stable_unpaved",
    "Unmatched_or_unknown": "unmatched_or_unknown",
}

def step7_change_class(sA, sB):
    if sA == "unpaved" and sB == "paved":
        return "Upgrade_unpaved_to_paved"
    if sA == "paved" and sB == "unpaved":
        return "Downgrade_paved_to_unpaved"
    if sA == "paved" and sB == "paved":
        return "Stable_paved"
    if sA == "unpaved" and sB == "unpaved":
        return "Stable_unpaved"
    return "Unmatched_or_unknown"

def step7_run_change_detection(OUT_STEP6: str, OUT_STEP7: str):
    print("\n" + "="*70)
    print(f"STEP 7 — CHANGE DETECTION ({YEAR_A} → {YEAR_B})")
    print("="*70)

    pred_a = os.path.join(OUT_STEP6, f"segments_{YEAR_A}_predicted.gpkg")
    pred_b = os.path.join(OUT_STEP6, f"segments_{YEAR_B}_predicted.gpkg")

    gA = gpd.read_file(pred_a, layer="predicted").to_crs(METRIC_CRS)
    gB = gpd.read_file(pred_b, layer="predicted").to_crs(METRIC_CRS)

    gB_small = gB[["seg_id", "pred_surface", "p_paved", "geometry"]].copy().rename(columns={
        "seg_id": f"seg_id_{YEAR_B}",
        "pred_surface": f"surface_{YEAR_B}",
        "p_paved": f"p_paved_{YEAR_B}",
    })

    joined = gpd.sjoin_nearest(
        gA, gB_small,
        how="left",
        max_distance=MAX_MATCH_DIST_M,
        distance_col="match_dist_m"
    )

    joined = joined.rename(columns={
        "pred_surface": f"surface_{YEAR_A}",
        "p_paved": f"p_paved_{YEAR_A}",
    })

    joined["change_class"] = joined.apply(
        lambda r: step7_change_class(r.get(f"surface_{YEAR_A}"), r.get(f"surface_{YEAR_B}")),
        axis=1
    )
    joined["len_km"] = joined.geometry.length / 1000.0

    out_csv = os.path.join(OUT_STEP7, f"surface_change_summary_{YEAR_A}_{YEAR_B}.csv")
    summary = (joined.groupby("change_class")["len_km"].sum().reset_index().sort_values("len_km", ascending=False))
    summary["len_km"] = summary["len_km"].round(3)
    summary.to_csv(out_csv, index=False)

    out_gpkg = os.path.join(OUT_STEP7, f"surface_change_{YEAR_A}_{YEAR_B}.gpkg")
    joined.to_file(out_gpkg, layer="change_all", driver="GPKG")

    for cls, layer_name in LAYER_BY_CLASS.items():
        sub = joined[joined["change_class"] == cls].copy()
        if len(sub) > 0:
            sub.to_file(out_gpkg, layer=layer_name, driver="GPKG")

    print("Saved change GPKG:", out_gpkg)
    print("Saved summary CSV:", out_csv)
    print(summary.to_string(index=False))


# ============================================================
# STEP 8 — Compare experiments (NO penalties)
# ============================================================
def _read_row(eval_csv: str, fold_name: str) -> dict:
    if not os.path.exists(eval_csv):
        return {}
    df = pd.read_csv(eval_csv)
    row = df[df["fold"].astype(str).str.upper() == fold_name.upper()]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def read_eval_for_report(eval_csv: str) -> dict:
    if not os.path.exists(eval_csv):
        return {}

    df = pd.read_csv(eval_csv)
    if df.empty:
        return {}

    row = df.iloc[0]

    return {
        "threshold": row.get("best_threshold", np.nan),
        "balanced_accuracy": row.get("balanced_accuracy", np.nan),
        "mcc": row.get("mcc", np.nan),
        "tn": row.get("tn", np.nan),
        "fp": row.get("fp", np.nan),
        "fn": row.get("fn", np.nan),
        "tp": row.get("tp", np.nan),
        "tn_pct": row.get("tn_pct", np.nan),
        "fp_pct": row.get("fp_pct", np.nan),
        "fn_pct": row.get("fn_pct", np.nan),
        "tp_pct": row.get("tp_pct", np.nan),
    }


    def _pull(prefix, row):
        if not row:
            return
        out[f"{prefix}_t"] = float(row.get("best_threshold", np.nan))
        out[f"{prefix}_acc"] = float(row.get("accuracy@bestT", np.nan))
        out[f"{prefix}_f1"] = float(row.get("f1@bestT", np.nan))
        out[f"{prefix}_balacc"] = float(row.get("balacc@bestT", np.nan))
        out[f"{prefix}_mcc"] = float(row.get("mcc@bestT", np.nan))
        out[f"{prefix}_tpr"] = float(row.get("recall_tpr@bestT", np.nan))
        out[f"{prefix}_tnr"] = float(row.get("tnr@bestT", np.nan))
        out[f"{prefix}_pos_rate"] = float(row.get("pos_rate@bestT", np.nan))
        for k in ["cm_tn", "cm_fp", "cm_fn", "cm_tp"]:
            v = row.get(k, np.nan)
            out[f"{prefix}_{k}"] = int(float(v)) if pd.notna(v) else np.nan

    _pull("F1", best_f1)
    _pull("BALACC", best_ba)
    return out

def read_qc(qc_csv: str) -> dict:
    if not os.path.exists(qc_csv):
        return {}
    df = pd.read_csv(qc_csv)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

def read_change_summary(summary_csv: str) -> dict:
    if not os.path.exists(summary_csv):
        return {}
    df = pd.read_csv(summary_csv)
    if df.empty or "change_class" not in df.columns or "len_km" not in df.columns:
        return {}
    total = float(df["len_km"].sum())
    unmatched = float(df.loc[df["change_class"] == "Unmatched_or_unknown", "len_km"].sum()) \
        if "Unmatched_or_unknown" in set(df["change_class"]) else 0.0
    return {"unmatched_pct_step7": (unmatched / total * 100.0) if total > 0 else np.nan}

def safe_to_csv(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.replace(".csv", f"_{ts}.csv")
        df.to_csv(alt, index=False)
        return alt

def step8_compare_and_report(experiment_names):
    rows = []

    for exp in experiment_names:
        exp_root = os.path.join(BASE_DIR, "experiments", exp)

        sensor, model_type = exp.split("_")

        eval_a = os.path.join(
            exp_root, "step5_models", f"eval_{model_type}_{YEAR_A}.csv"
        )
        eval_b = os.path.join(
            exp_root, "step5_models", f"eval_{model_type}_{YEAR_B}.csv"
        )

        qc_a = os.path.join(exp_root, "step6_predictions", f"qc_{YEAR_A}.csv")
        qc_b = os.path.join(exp_root, "step6_predictions", f"qc_{YEAR_B}.csv")

        chg = os.path.join(exp_root, "step7_change_detection",
                           f"surface_change_summary_{YEAR_A}_{YEAR_B}.csv")

        eA = read_eval_for_report(eval_a)
        eB = read_eval_for_report(eval_b)
        qA = read_qc(qc_a)
        qB = read_qc(qc_b)
        cS = read_change_summary(chg)

        rows.append({
            "experiment": exp,

            # 2018
            "threshold_2018": eA.get("threshold", np.nan),
            "balacc_2018": eA.get("balanced_accuracy", np.nan),
            "mcc_2018": eA.get("mcc", np.nan),
            "tn_2018": eA.get("tn", np.nan),
            "fp_2018": eA.get("fp", np.nan),
            "fn_2018": eA.get("fn", np.nan),
            "tp_2018": eA.get("tp", np.nan),
            "tn_pct_2018": eA.get("tn_pct", np.nan),
            "fp_pct_2018": eA.get("fp_pct", np.nan),
            "fn_pct_2018": eA.get("fn_pct", np.nan),
            "tp_pct_2018": eA.get("tp_pct", np.nan),

            # 2024
            "threshold_2024": eB.get("threshold", np.nan),
            "balacc_2024": eB.get("balanced_accuracy", np.nan),
            "mcc_2024": eB.get("mcc", np.nan),
            "tn_2024": eB.get("tn", np.nan),
            "fp_2024": eB.get("fp", np.nan),
            "fn_2024": eB.get("fn", np.nan),
            "tp_2024": eB.get("tp", np.nan),
            "tn_pct_2024": eB.get("tn_pct", np.nan),
            "fp_pct_2024": eB.get("fp_pct", np.nan),
            "fn_pct_2024": eB.get("fn_pct", np.nan),
            "tp_pct_2024": eB.get("tp_pct", np.nan),
        })

    comp = pd.DataFrame(rows)

    out_dir = os.path.join(BASE_DIR, "experiments", "_COMPARISON_STEP8")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"comparison_step8_{YEAR_A}_{YEAR_B}.csv")
    saved_path = safe_to_csv(comp, out_csv)

    print("\n" + "="*120)
    print("STEP 8 — EXPERIMENT COMPARISON TABLE (NO penalties; includes MCC/TPR/TNR + CM fields)")
    print("="*120)
    print(comp.to_string(index=False))
    print(f"\nSaved comparison CSV: {saved_path}")


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def run_experiment(experiment_name: str):
    sensor, model_type = experiment_name.split("_")

    OUT = make_experiment_dirs(experiment_name)

    print("\n" + "#"*90)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print("#"*90)


    if RUN_STEP_4:

        GLOBAL_DIR = os.path.join(BASE_DIR, "GLOBAL_preprocessing")

        seg_all_a = os.path.join(GLOBAL_DIR, f"segments_{YEAR_A}_all.gpkg")
        seg_all_b = os.path.join(GLOBAL_DIR, f"segments_{YEAR_B}_all.gpkg")

        seg_train_a = os.path.join(GLOBAL_DIR, f"segments_{YEAR_A}_train.gpkg")
        seg_train_b = os.path.join(GLOBAL_DIR, f"segments_{YEAR_B}_train.gpkg")

        if sensor == "L8":
            raster_a = os.path.join(IMAGERY_DIR, f"L8_{YEAR_A}.tif")
            raster_b = os.path.join(IMAGERY_DIR, f"L8_{YEAR_B}.tif")

        elif sensor == "S2":
            raster_a = os.path.join(IMAGERY_DIR, f"S2_{YEAR_A}.tif")
            raster_b = os.path.join(IMAGERY_DIR, f"S2_{YEAR_B}.tif")

        elif sensor == "S1":
            raster_a = os.path.join(IMAGERY_DIR, f"S1_{YEAR_A}.tif")
            raster_b = os.path.join(IMAGERY_DIR, f"S1_{YEAR_B}.tif")

        else:
            raise ValueError("Unknown sensor")

        out_a = os.path.join(OUT["OUT_STEP4_MERGED"], f"X_{YEAR_A}.csv")
        out_b = os.path.join(OUT["OUT_STEP4_MERGED"], f"X_{YEAR_B}.csv")

        step4_extract_local(seg_all_a, raster_a, sensor, out_a)
        step4_extract_local(seg_all_b, raster_b, sensor, out_b)

    if RUN_STEP_5_6:

        GLOBAL_DIR = os.path.join(BASE_DIR, "GLOBAL_preprocessing")

        seg_all_a = os.path.join(GLOBAL_DIR, f"segments_{YEAR_A}_all.gpkg")
        seg_all_b = os.path.join(GLOBAL_DIR, f"segments_{YEAR_B}_all.gpkg")

        seg_train_a = os.path.join(GLOBAL_DIR, f"segments_{YEAR_A}_train.gpkg")
        seg_train_b = os.path.join(GLOBAL_DIR, f"segments_{YEAR_B}_train.gpkg")

        X_a = os.path.join(OUT["OUT_STEP4_MERGED"], f"X_{YEAR_A}.csv")
        X_b = os.path.join(OUT["OUT_STEP4_MERGED"], f"X_{YEAR_B}.csv")

        for p in [seg_all_a, seg_all_b, seg_train_a, seg_train_b, X_a, X_b]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required file for Step 5/6:\n{p}")

        model_a = os.path.join(OUT["OUT_STEP5"], f"model_{model_type}_{YEAR_A}.joblib")
        model_b = os.path.join(OUT["OUT_STEP5"], f"model_{model_type}_{YEAR_B}.joblib")

        eval_a = os.path.join(OUT["OUT_STEP5"], f"eval_{model_type}_{YEAR_A}.csv")
        eval_b = os.path.join(OUT["OUT_STEP5"], f"eval_{model_type}_{YEAR_B}.csv")

        step5_train_model_with_validation(
            YEAR_A, seg_train_a, X_a, model_a, eval_a, model_type
        )
        step5_train_model_with_validation(
            YEAR_B, seg_train_b, X_b, model_b, eval_b, model_type
        )

        step6_predict_all_with_qc(
            YEAR_A, seg_all_a, X_a, model_a,
            os.path.join(OUT["OUT_STEP6"], f"segments_{YEAR_A}_predicted.gpkg"),
            os.path.join(OUT["OUT_STEP6"], f"qc_{YEAR_A}.csv")
        )
        step6_predict_all_with_qc(
            YEAR_B, seg_all_b, X_b, model_b,
            os.path.join(OUT["OUT_STEP6"], f"segments_{YEAR_B}_predicted.gpkg"),
            os.path.join(OUT["OUT_STEP6"], f"qc_{YEAR_B}.csv")
        )

    if RUN_STEP_7:
        step7_run_change_detection(OUT["OUT_STEP6"], OUT["OUT_STEP7"])


# ============================================================
# MAIN
# ============================================================
# ============================================================
# RUN STEP 1–3 ONCE ONLY
# ============================================================

GLOBAL_DIR = os.path.join(BASE_DIR, "GLOBAL_preprocessing")
os.makedirs(GLOBAL_DIR, exist_ok=True)

if RUN_STEP_1:
    step1_extract_unique_surfaces(SNAP_A, YEAR_A, GLOBAL_DIR)
    step1_extract_unique_surfaces(SNAP_B, YEAR_B, GLOBAL_DIR)

if RUN_STEP_2:
    step2_label_snapshot(SNAP_A, YEAR_A, GLOBAL_DIR)
    step2_label_snapshot(SNAP_B, YEAR_B, GLOBAL_DIR)

if RUN_STEP_3:
    seg_all_a = step3_segment_all_roads(SNAP_A, YEAR_A, GLOBAL_DIR)
    seg_all_b = step3_segment_all_roads(SNAP_B, YEAR_B, GLOBAL_DIR)

    train_a = os.path.join(GLOBAL_DIR, f"train_{YEAR_A}.gpkg")
    train_b = os.path.join(GLOBAL_DIR, f"train_{YEAR_B}.gpkg")

    step3_make_train_segments(seg_all_a, train_a, YEAR_A, GLOBAL_DIR)
    step3_make_train_segments(seg_all_b, train_b, YEAR_B, GLOBAL_DIR)


if __name__ == "__main__":
    for exp in RUN_EXPERIMENTS:
        for model_type in MODEL_TYPES:
            run_experiment(f"{exp}_{model_type}")

    if RUN_STEP_8:
        FULL_EXPERIMENTS = [
            f"{sensor}_{model}"
            for sensor in RUN_EXPERIMENTS
            for model in MODEL_TYPES
        ]

        step8_compare_and_report(FULL_EXPERIMENTS)

# ============================================================
# STEP 9 — FINAL SIMPLE COMPARISON EXCEL
# ============================================================

IN_CSV = os.path.join(
    BASE_DIR,
    "experiments",
    "_COMPARISON_STEP8",
    f"comparison_step8_{YEAR_A}_{YEAR_B}.csv"
)

OUT_XLSX = os.path.join(
    BASE_DIR,
    "experiments",
    "_COMPARISON_STEP8",
    "experiment_comparison_results.xlsx"
)

df = pd.read_csv(IN_CSV)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="COMPARISON", index=False)

print("✅ FINAL EXCEL CREATED:")
print(OUT_XLSX)
