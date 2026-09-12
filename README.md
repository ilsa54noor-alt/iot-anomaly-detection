# IoT Anomaly Detection
A machine learning project applying Random Forest classification to detect
anomalous (attack) traffic in IoT networks, using the CICIoT2023 dataset.

## Motivation

IoT networks are increasingly targeted by attackers, and traditional
signature-based security tools often struggle to keep up with new or
varied attack behavior. This project explores whether a standard machine
learning classifier can distinguish normal IoT traffic from several
distinct categories of attack traffic, and evaluates where such a model
succeeds and where it struggles.

## Dataset

[CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
(Canadian Institute for Cybersecurity, University of New Brunswick) —
a large-scale IoT attack dataset covering 33 attacks across 7 categories.

Due to project scope and timeline, four traffic types were used:

| File | Type |
|---|---|
| BenignTraffic.pcap.csv | Normal traffic (baseline) |
| DDoS-SYN_Flood.pcap.csv | Volumetric flood attack |
| Mirai-udpplain.pcap.csv | IoT-specific botnet attack |
| Recon-PortScan.pcap.csv | Reconnaissance / stealth scanning |

These four were chosen to cover meaningfully different attack behaviors
(a flood, a botnet-driven flood, and a quiet reconnaissance scan) rather
than several variants of the same attack type.

## Method

1. **Data loading & labeling** — loaded the four CSVs, labeled each by
   source file, and combined into a single dataset (747,150 rows).
2. **Cleaning** — removed 93,583 duplicate rows, 48 rows with missing
   values, and 27 rows with infinite values in the `Rate` feature.
   Final dataset: 653,540 rows.
3. **Split** — 80/20 train/test split, stratified by label to preserve
   class proportions given the dataset's class imbalance.
4. **Model** — Random Forest Classifier (100 trees). No feature scaling
   applied, since Random Forest's threshold-based splits are unaffected
   by differing feature ranges.
5. **Evaluation** — precision, recall, F1-score per class, and a
   confusion matrix, rather than relying on overall accuracy alone.

## Results

Overall accuracy: **93.2%**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Benign | 0.90 | 0.99 | 0.94 |
| DDoS-SYN_Flood | 1.00 | 1.00 | 1.00 |
| Mirai-udpplain | 1.00 | 1.00 | 1.00 |
| Recon-PortScan | 0.89 | 0.51 | 0.65 |

### Key finding

While overall accuracy looks strong, **Recon-PortScan recall was only
0.51** — the model missed roughly half of all actual port-scan attempts,
with 7,826 of 16,049 misclassified as benign traffic. DDoS-SYN_Flood and
Mirai-udpplain, by contrast, were classified almost perfectly.

This gap makes sense given the nature of the attacks: flooding attacks
(SYN Flood, Mirai) produce high-volume, clearly abnormal traffic
patterns, while a port scan is deliberately quiet — one connection at a
time — making it resemble low-volume normal activity far more closely.
This is a known challenge in network intrusion detection, and it
demonstrates why accuracy alone is an insufficient metric for imbalanced,
security-relevant classification tasks: a high overall score can mask a
meaningful operational blind spot against a specific, realistic attack
type.

## Limitations

- Only 4 of the dataset's 33 available attack types were used.
- Single train/test split — no cross-validation performed.
- Default Random Forest hyperparameters — no tuning conducted.

## Tech stack

Python, pandas, scikit-learn, JupyterLab

## Files

- `iot_anomaly_detection.ipynb` — full pipeline: loading, cleaning,
  splitting, training, evaluation.
- `notes.md` — detailed working notes and reasoning for each step.
