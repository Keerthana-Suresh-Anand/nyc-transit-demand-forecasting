"""
Population Stability Index (PSI) drift detection.

Exposes ``compute_psi``, used by the monitoring pipeline to score weather-feature
drift for the dashboard/report. PSI is **informational only** — retraining is
triggered solely by MAE degradation (see ``monitor_performance``), never by PSI,
because PSI on weather features fires seasonal false alarms every spring/fall.
This module deliberately does not write a retrain flag.
"""
import numpy as np


def compute_psi(reference: np.ndarray, recent: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between reference and recent distributions."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    rec_counts = np.histogram(recent, bins=breakpoints)[0]

    ref_pct = (ref_counts + 1e-8) / len(reference)
    rec_pct = (rec_counts + 1e-8) / len(recent)

    psi = np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct))
    return float(psi)
