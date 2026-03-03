import os
import pandas as pd
from scipy import stats

# =========================
# 0) PATHS
# =========================
BASE_DIR = "./data/findings"
BLIND_CSV = os.path.join(BASE_DIR, "Blindspots_named_FINAL.csv")
NONBLIND_CSV = os.path.join(BASE_DIR, "non_blindspots.csv")
OUT_XLSX = os.path.join(BASE_DIR, "analysis_FINAL_with_districts.xlsx")

VALUE_COL = "built_mean"
DISTRICT_COL = "District"  # change only if your column name differs

# =========================
# 1) LOAD
# =========================
blind = pd.read_csv(BLIND_CSV)
non = pd.read_csv(NONBLIND_CSV)

# Strip column name spaces (VERY IMPORTANT)
blind.columns = blind.columns.str.strip()
non.columns = non.columns.str.strip()

print("\n--- COLUMNS ---")
print("Blind:", list(blind.columns))
print("Non:", list(non.columns))

# Check built_mean exists
if VALUE_COL not in blind.columns:
    raise KeyError(f"'{VALUE_COL}' not found in blind CSV. Available: {list(blind.columns)}")
if VALUE_COL not in non.columns:
    raise KeyError(f"'{VALUE_COL}' not found in non-blind CSV. Available: {list(non.columns)}")

# Convert to numeric safely
blind[VALUE_COL] = pd.to_numeric(blind[VALUE_COL], errors="coerce")
non[VALUE_COL] = pd.to_numeric(non[VALUE_COL], errors="coerce")

print("\n--- NA COUNTS AFTER NUMERIC CONVERSION ---")
print("Blind NaNs:", int(blind[VALUE_COL].isna().sum()), "of", len(blind))
print("Non NaNs:", int(non[VALUE_COL].isna().sum()), "of", len(non))

# Drop NaNs
blind = blind.dropna(subset=[VALUE_COL]).copy()
non = non.dropna(subset=[VALUE_COL]).copy()

print("\n--- ROW COUNTS AFTER DROPPING NaNs ---")
print("Blind rows:", len(blind))
print("Non rows:", len(non))

# Assign groups
blind["group"] = "Blindspots"
non["group"] = "Non-blindspots"
df = pd.concat([blind, non], ignore_index=True)

# Create x and y
x = df[df["group"] == "Blindspots"][VALUE_COL]
y = df[df["group"] == "Non-blindspots"][VALUE_COL]

print("\n--- FINAL SAMPLE SIZES FOR TESTS ---")
print("x (Blindspots):", len(x))
print("y (Non-blindspots):", len(y))

# Stop early if empty (prevents your error)
if len(x) == 0 or len(y) == 0:
    raise ValueError(
        "One group has ZERO values after cleaning.\n"
        "Fix: check that both CSVs contain numeric 'built_mean' values.\n"
        "Also check if built_mean column is stored as text with commas or weird symbols."
    )

# =========================
# 2) DESCRIPTIVE STATS
# =========================
desc = df.groupby("group")[VALUE_COL].describe()

# =========================
# 3) STAT TESTS
# =========================
mw = stats.mannwhitneyu(x, y, alternative="two-sided")
t, p = stats.ttest_ind(x, y, equal_var=False)

tests = pd.DataFrame({
    "Test": ["Mann-Whitney U (two-sided)", "Welch t-test"],
    "Result": [f"U={mw.statistic:.2f}, p={mw.pvalue:.4g}", f"t={t:.2f}, p={p:.4g}"]
})

# =========================
# 4) DISTRICT SUMMARY (from blind only)
# =========================
blind.columns = blind.columns.str.strip()
if DISTRICT_COL not in blind.columns:
    print("\n⚠️ District column not found, skipping district summary.")
    district_summary = pd.DataFrame()
else:
    district_summary = (blind.groupby(DISTRICT_COL)
                        .agg(blindspot_count=(DISTRICT_COL, "count"),
                             mean_built=(VALUE_COL, "mean"))
                        .reset_index()
                        .sort_values("blindspot_count", ascending=False))

# =========================
# 5) EXPORT EXCEL
# =========================
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    desc.to_excel(writer, sheet_name="Descriptive")
    tests.to_excel(writer, sheet_name="Tests", index=False)

    if not district_summary.empty:
        district_summary.to_excel(writer, sheet_name="Districts", index=False)

print("\n✅ DONE. Saved:", OUT_XLSX)