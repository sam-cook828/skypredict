# Flight Delay Predictions

## Overview

Flight delays don't just inconvenience passengers — they cascade across entire airline networks, costing the U.S. aviation industry billions of dollars each year. This project builds a **two-stage machine learning pipeline** that predicts both *whether* a flight will be delayed and *how long* that delay will last, using only information available **2 hours before departure**.

- **Stage 1 (Classification):** Predicts whether a flight will depart 15+ minutes late (`DEP_DEL15`)
- **Stage 2 (Regression):** Estimates the actual delay in minutes (`DEP_DELAY`) for flagged flights

---

## Dataset

| Source | Description |
|--------|-------------|
| **BTS (Bureau of Transportation Statistics)** | Every domestic U.S. flight, 2015–2019 (31.6M records, 214 columns) |
| **NOAA** | Hourly weather observations joined to origin airports |

**Train/Test Split:** 2015–2018 for training (~23.9M rows), 2019 as a blind test set (~7.3M rows).

Cancelled flights and null-target records are excluded, leaving **31,197,330 ML-ready rows**.

---

## Feature Engineering

The final feature vector is **~813 dimensions** (sparse), built from 6 families of engineered features — all using only pre-departure information:

| Family | Count | Examples |
|--------|-------|---------|
| Schedule / Calendar | 7 | `CRS_DEP_TIME`, `DISTANCE`, `DAY_OF_WEEK` |
| Raw Weather (NOAA) | 12 | `HourlyVisibility`, `HourlyWindSpeed`, `HourlyPrecipitation` |
| Engineered Time | 8 | `hour_sin/cos`, `month_sin/cos`, `is_weekend` |
| Weather Composites | 5 | `weather_severity`, `low_visibility`, `has_precipitation` |
| Congestion & Rolling | 4 | `origin_hourly_flights`, `tail_rolling_delay_rate` |
| Target Encoding | 4 | `carrier_te`, `origin_te`, `route_delay_rate` |
| Multi-Timeframe RFM | 12 | 7d/14d/30d rolling delay rates per carrier, origin, dest, route |
| Graph / Network | 5 | `origin_pagerank`, `dest_pagerank`, `total_degree`, `carrier_pagerank` |
| Event-Based | 3 | `is_holiday`, `is_national_event`, `holiday_proximity` |
| Categorical (OHE) | ~753 | `ORIGIN`, `DEST`, `OP_UNIQUE_CARRIER`, `DEP_TIME_BLK` |

**Key insight from EDA:** Scheduling and carrier factors dominate delay prediction, while individual weather features have weak linear correlations (|r| < 0.09). Weather composites and tail/congestion rolling features capture the non-linear signal that tree-based models exploit.

---

## Models

### Stage 1: Binary Classification

| Model | Architecture | AUC-PR | AUC-ROC | F1 | Precision | Recall |
|-------|-------------|--------|---------|-----|-----------|--------|
| MLP-A (Shallow) | [813→64→2] | 0.4995 | 0.7854 | 0.7665 | 0.8168 | 0.7435 |
| MLP-B (Wider) | [813→128→2] | 0.5054 | 0.7859 | 0.7868 | 0.8183 | 0.7701 |
| MLP-C (Deeper) | [813→128→64→2] | 0.5084 | 0.7845 | 0.7834 | 0.8175 | 0.7657 |
| MLP-D (No OHE) | [60→128→64→2] | 0.4915 | 0.7842 | 0.7844 | 0.8184 | 0.7668 |
| **GBT (Tuned)** | **100 trees, depth 7, lr=0.1** | **0.5570** | **0.8045** | **0.7912** | **0.8254** | **0.7740** |

**Best Stage 1:** GBT (AUC-PR = 0.557, ~2.9× lift over random baseline of 0.18)

### Stage 2: Delay Duration Regression

| Model | Target Transform | MAPE | RMSE | MAE |
|-------|-----------------|------|------|-----|
| MLP-A (Shallow) | Clip@P99 | 101.50% | 63.23 | 44.71 |
| **MLP-B (Deeper)** | **Clip@P99** | **61.62%** | **74.00** | **46.36** |
| GBT-Reg (Clip@P99) | Clip@P99 | 90.50% | 73.36 | 48.54 |
| GBT-Reg (No Clip) | Raw target | 95.00% | 86.22 | 52.13 |
| GBT-Reg (Log) | log(delay+1) | 65.85% | 89.53 | 44.43 |

**Best Stage 2:** PyTorch MLP-B Deeper [813→128→64→1] (MAPE = 61.62%, 31pp improvement over Phase 2 baseline of 92.59%)

---

## Phase-over-Phase Improvement

| Metric | Phase 2 Baseline | Phase 3 Best | Change |
|--------|-----------------|--------------|--------|
| Training data | 5.8M flights | 23.9M flights | 4.1× more |
| Stage 1 AUC-PR | 0.5265 | 0.5570 | +5.8% relative |
| Stage 2 MAPE | 92.59% | 61.62% | −31.0 pp |

---

## Pipeline Architecture

```
Raw OTPW 60M Dataset (2015-2019)
        ↓
Data Ingestion & Filtering
(Remove cancelled/null-target flights)
        ↓
Feature Engineering Sub-pipeline
(Weather imputation, cyclic encoding, rolling aggregates,
 target encoding, graph PageRank, event flags,
 OHE → VectorAssembler → StandardScaler)
        ↓ (~813-dim sparse vector)
┌─────────────────────────────────┐
│  Stage 1: GBT Classifier        │
│  Predicts DEP_DEL15 (binary)    │
│  Expanding-window temporal CV   │
└─────────────────────────────────┘
        ↓ (predicted delayed flights)
┌─────────────────────────────────┐
│  Stage 2: MLP-B Regressor      │
│  Predicts DEP_DELAY (minutes)   │
│  PyTorch, OOF training approach │
└─────────────────────────────────┘
```

**Key design decisions:**
- **Temporal train/test split** (year-based, not random) prevents future data leakage
- **Expanding-window CV** (2015→2016, 2015-16→2017, 2015-17→2018) mimics production deployment
- **Out-of-fold (OOF) training** for Stage 2 ensures the regressor is trained on realistic, noisy classifier outputs rather than in-sample predictions
- **Clip@P99 target transform** for MLP regression removes extreme outliers (>359 min) without discarding 99% of the distribution

---

## Key Findings

1. **GBT outperforms MLP for classification** on this tabular dataset — consistent with the broader ML literature on structured data. GBT natively handles sparse high-cardinality features and threshold interactions.

2. **MLP outperforms GBT for regression** — the smooth nonlinear mapping learned by a neural network better captures the continuous nature of delay duration than GBT's discrete leaf averaging.

3. **Dense numeric features carry most of the signal.** The MLP-D experiment (60 features, no OHE) achieved F1 = 0.784 vs. MLP-C's 0.783, confirming that the 753 OHE dimensions add little incremental value when graph and target-encoding features are present.

4. **Target transformation is critical for GBT regression.** Log-transforming the skewed delay distribution reduced GBT MAPE from 90.5% (clip) to 65.9%, a 25pp gain from a single preprocessing change.

5. **Signal ceiling at ~0.80 AUC-ROC.** Both GBT and MLP converge near this value, reflecting the inherent unpredictability of events occurring after the 2-hour prediction window (runway incidents, late inbound aircraft, ATC ground stops).

---

## Compute Environment

| Component | Specification |
|-----------|--------------|
| Driver | m6g.4xlarge (16 vCPU, 64 GB RAM) |
| Workers | 7 × m6g.xlarge (4 vCPU, 16 GB RAM each) |
| Databricks Runtime | 17.3 LTS |
| Total wall time | ~16 hours (Stage 1 + Stage 2 + OOF) |

---

## Repository Structure

```
├── notebooks/
│   ├── Team_1_1_Phase_3_Report.html   # Full Databricks report
│   └── ...                             # Source notebooks
├── README.md
└── ...
```

---

## Glossary

| Term | Definition |
|------|-----------|
| BTS | Bureau of Transportation Statistics |
| NOAA | National Oceanic and Atmospheric Administration |
| GBT | Gradient Boosted Trees |
| MLP | Multilayer Perceptron (neural network) |
| AUC-PR | Area Under the Precision-Recall Curve (primary classification metric) |
| MAPE | Mean Absolute Percentage Error (primary regression metric) |
| OHE | One-Hot Encoding |
| OOF | Out-of-Fold |
| PageRank | Graph algorithm ranking airport importance by flight connections |
| P99 | 99th percentile clip threshold for extreme delay outliers |

---

## Limitations & Next Steps

- **Signal ceiling:** Many delays are caused by events that emerge after the 2-hour prediction window. Integrating real-time crew scheduling, inbound aircraft tracking, and gate-level congestion data would address this.
- **Mild leakage:** PageRank features were computed on the aggregate 2015–2018 graph rather than in a rolling/lagged manner. The StandardScaler was also fit before strict train/test separation. Future work should correct both.
- **No uncertainty quantification:** Conformal prediction or quantile regression would add calibrated prediction intervals to Stage 2 outputs.
- **OHE dimensionality:** Entity embeddings or feature hashing could replace the 753 OHE columns with a denser, more MLP-friendly representation.
