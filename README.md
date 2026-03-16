# Fitness Tracker ML

> **Wrist sensor → exercise classifier + rep counter.** Wear a MetaMotion IMU while lifting. This project figures out what exercise you're doing and counts your reps — automatically.

---

## What it does

| Input | Output |
|-------|--------|
| Raw accelerometer + gyroscope CSV files | Exercise label (bench / squat / ohp / row / deadlift) |
| 5 participants, multiple sets each | Rep count per set |
| 12.5 Hz acc · 25 Hz gyro | MAE ~1 rep · Random Forest classifier |

---

## Pipeline — 6 steps

```
Raw CSVs  →  [1] Load & clean  →  01_data_processed.pkl
                                         ↓
                                       [2] Visualise (EDA)
                                                   ↓
                                            [3] Remove outliers  →  02_outliers_removed.pkl
                                                                  ↓
                                                         [4] Feature engineering  →  03_features_extracted.pkl
                                                                                              ↓
                                                                                     [5] Train model  →  Random Forest ✓
                                                                                                     ↓
                                                                                           [6] Count reps  →  MAE evaluation
                                  (back to step 1 file — needs clean signal)
```

| Step | Script | What happens |
|------|--------|-------------|
| 1 | `make_dataset.py` | Parse filenames, split acc/gyro, merge, resample to 200ms |
| 2 | `visualize.py` | Plot signals per exercise/participant, export PNGs |
| 3 | `remove_outliers.py` | Chauvenet's criterion per label → replace outliers with NaN |
| 4 | `build_features.py` | LPF · PCA · magnitude · rolling stats · FFT · K-Means |
| 5 | `train_model.py` | 5 models × 5 feature sets, gridsearch, confusion matrix |
| 6 | `count_repetitions.py` | Butterworth filter + peak detection → rep count |

---

## Key techniques

**Outlier removal** — Chauvenet's criterion flags values with probability < 1/(2N) under a normal distribution. Runs per exercise label, not globally, because a heavy deadlift signal looks nothing like an overhead press.

**Feature engineering** — 6 raw axes become 100+ features: Butterworth low-pass filter (1 Hz cutoff), PCA (3 components), vector magnitudes (acc_r, gyro_r), rolling mean/std (1s window), FFT frequency bins (2.8s window), K-Means cluster label (k=5).

**Classification** — Random Forest won across all 5 feature sets. Validated two ways: random 80/20 split and leave-one-participant-out (LOPO) to test true generalisation.

**Rep counting** — no ML needed here. Each rep = one cycle in the filtered signal. `scipy.signal.argrelextrema` finds local maxima. Cutoff frequency tuned per exercise (rows use gyro_x instead of acc_r because the motion is rotational).

---

> ML4QS = open-source library from *Machine Learning for the Quantified Self* (Hoogendoorn & Funk, 2018)

---

## Run it

```bash
pip install pandas numpy matplotlib scipy scikit-learn seaborn

python src/data/make_dataset.py
python src/visualization/visualize.py
python src/features/remove_outliers.py
python src/features/build_features.py
python src/models/train_model.py
python src/models/count_repetitions.py
```

Raw data goes in `data/raw/MetaMotion/`. Interim pickles are generated automatically in order.

---

## Results

- **Exercise classification** — Random Forest, full feature set. LOPO accuracy lower than random-split accuracy → model partially learns participant-specific movement patterns, not just exercise physics.
- **Rep counting** — average error ~1 rep per set across all exercises and weight categories.

<img width="2901" height="1641" alt="tech_features" src="https://github.com/user-attachments/assets/9537e114-8561-4df7-b3e3-36d57defeffa" />

  <img width="2522" height="3621" alt="pipeline" src="https://github.com/user-attachments/assets/94bd0f73-3769-4217-a983-a7059e1a0049" />

<img width="3251" height="1810" alt="models_reps" src="https://github.com/user-attachments/assets/e718c429-0744-48ce-b6bb-8640f75a6631" />
