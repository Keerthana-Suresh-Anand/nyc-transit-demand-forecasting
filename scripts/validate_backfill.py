"""
Post-backfill validation. Run after rebuild_silver + run_training to confirm
the expanded 2022-2025 dataset is clean before retraining.

Checks:
  1. Missing dates in the daily series
  2. Structural break at the 2024->2025 dataset join
  3. ADF + KPSS stationarity tests
  4. ACF/PACF plots (saved to reports/)
  5. PSI on weather features across sub-periods
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

load_dotenv()

from src.utils.config import GOLD_SARIMA_LOCAL_PATH, S3_GOLD_SARIMA_KEY
from src.utils.s3_helpers import download_s3_file, get_s3_client

warnings.filterwarnings("ignore")

REPORTS_DIR = Path("reports/validation")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def result(label: str, status: str, detail: str = "") -> None:
    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}.get(status, "?")
    print(f"  [{icon}] {label}: {status}  {detail}")


# ── Load gold data ─────────────────────────────────────────────────────────────
section("Loading data")
if not GOLD_SARIMA_LOCAL_PATH.exists():
    print(f"  Gold not found locally — downloading from S3...")
    GOLD_SARIMA_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    found = download_s3_file(s3, S3_GOLD_SARIMA_KEY, GOLD_SARIMA_LOCAL_PATH)
    if not found:
        print("  Gold file not found in S3 either. Run the training pipeline first.")
        raise SystemExit(1)

df = pd.read_parquet(GOLD_SARIMA_LOCAL_PATH)
df.index = pd.to_datetime(df.index)
df = df.sort_index()
ridership = df["daily_ridership"]

print(f"  Date range : {df.index.min().date()} to {df.index.max().date()}")
print(f"  Total rows : {len(df):,}")

# ── 1. Missing dates ───────────────────────────────────────────────────────────
section("1. Missing dates")
full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
missing = full_range.difference(df.index)
if missing.empty:
    result("No missing dates", PASS)
else:
    result(f"{len(missing)} missing dates", WARN, f"first: {missing[0].date()}, last: {missing[-1].date()}")
    print(f"     {list(missing[:10].date)}")

# ── 2. Structural break at 2024→2025 join ─────────────────────────────────────
section("2. Structural break at 2024→2025 dataset join")
pre  = ridership["2024-10-01":"2024-12-31"].dropna()
post = ridership["2025-01-01":"2025-03-31"].dropna()

if len(pre) > 0 and len(post) > 0:
    t_stat, p_value = stats.ttest_ind(pre, post)
    pre_mean  = pre.mean() / 1e6
    post_mean = post.mean() / 1e6
    pct_diff  = (post_mean - pre_mean) / pre_mean * 100
    print(f"  Q4 2024 mean : {pre_mean:.3f}M riders/day")
    print(f"  Q1 2025 mean : {post_mean:.3f}M riders/day")
    print(f"  Difference   : {pct_diff:+.1f}%")
    if p_value < 0.05 and abs(pct_diff) > 15:
        result("Level shift at join", WARN,
               f"t={t_stat:.2f}, p={p_value:.4f} — review ridership plot")
    else:
        result("No significant level shift at join", PASS,
               f"t={t_stat:.2f}, p={p_value:.4f}")
else:
    result("Insufficient data around join", WARN)

# ── 3. Stationarity ────────────────────────────────────────────────────────────
section("3. Stationarity (ADF + KPSS on differenced series)")
differenced = ridership.diff(1).dropna()

adf_stat, adf_p, _, _, adf_crit, _ = adfuller(differenced, autolag="AIC")
adf_status = PASS if adf_p < 0.05 else FAIL
result("ADF test (H0: unit root)", adf_status,
       f"stat={adf_stat:.3f}, p={adf_p:.4f} — {'stationary' if adf_p < 0.05 else 'non-stationary'}")

kpss_stat, kpss_p, _, kpss_crit = kpss(differenced, regression="c", nlags="auto")
kpss_status = PASS if kpss_p > 0.05 else FAIL
result("KPSS test (H0: stationary)", kpss_status,
       f"stat={kpss_stat:.3f}, p={kpss_p:.4f} — {'stationary' if kpss_p > 0.05 else 'non-stationary'}")

# ── 4. ACF / PACF ─────────────────────────────────────────────────────────────
section("4. ACF / PACF plots")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(differenced, lags=28, ax=axes[0], title="ACF — First-differenced ridership (lags 0–28)")
plot_pacf(differenced, lags=28, ax=axes[1], title="PACF — First-differenced ridership (lags 0–28)")
plt.tight_layout()
acf_path = REPORTS_DIR / "acf_pacf.png"
plt.savefig(acf_path)
plt.close()
result("ACF/PACF plot saved", PASS, str(acf_path))

# ── 5. Ridership plot across full range ───────────────────────────────────────
section("5. Ridership plot (full range)")
fig, ax = plt.subplots(figsize=(14, 4))
ridership.plot(ax=ax, linewidth=0.8, color="steelblue")
ax.axvline(pd.Timestamp("2025-01-01"), color="red", linestyle="--", linewidth=1, label="2024→2025 join")
ax.set_title("Daily ridership 2022–2025 (check for level shifts or anomalies)")
ax.set_ylabel("Riders")
ax.legend()
plt.tight_layout()
plot_path = REPORTS_DIR / "ridership_full_range.png"
plt.savefig(plot_path)
plt.close()
result("Ridership plot saved", PASS, str(plot_path))

# ── 6. PSI on weather features across sub-periods ─────────────────────────────
section("6. PSI — weather feature distributions across sub-periods")

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.linspace(0, 100, bins + 1)
    expected_pcts = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / len(expected)
    actual_pcts   = np.histogram(actual,   bins=np.percentile(expected, breakpoints))[0] / len(actual)
    expected_pcts = np.where(expected_pcts == 0, 1e-6, expected_pcts)
    actual_pcts   = np.where(actual_pcts   == 0, 1e-6, actual_pcts)
    return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))

periods = {
    "2022": df["2022-01-01":"2022-12-31"],
    "2023": df["2023-01-01":"2023-12-31"],
    "2024": df["2024-01-01":"2024-12-31"],
    "2025": df["2025-01-01":],
}
reference = periods["2022"]

for feature in ["temp", "precip", "snow"]:
    if feature not in df.columns:
        continue
    print(f"\n  {feature}:")
    ref_vals = reference[feature].dropna().values
    for period_name, period_df in list(periods.items())[1:]:
        vals = period_df[feature].dropna().values
        if len(vals) == 0:
            continue
        psi = compute_psi(ref_vals, vals)
        status = PASS if psi < 0.2 else WARN if psi < 0.5 else FAIL
        result(f"  PSI 2022 vs {period_name}", status, f"{psi:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
print("  Review the plots in reports/validation/ before retraining.")
print("  Key things to check:")
print("    - ridership_full_range.png : no unexpected level shifts or gaps")
print("    - acf_pacf.png             : lag-7 spike confirms weekly seasonality")
print("    - Structural break result  : >15% level shift at join warrants investigation")
print("    - ADF + KPSS both PASS     : series is stationary after first differencing")
