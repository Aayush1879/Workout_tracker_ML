import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# Load data
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")
predict_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Dealing with missing values (imputation)
df.info()
for col in predict_columns:
    df[col] = df[col].interpolate()

df.info()

# Calculating set duration
duration = df[df["Set"] == 1].index[-1] - df[df["Set"] == 1].index[0]
print(f"Duration of set 1: {duration} seconds")

df["Set"].unique()
for s in df["Set"].unique():
    duration = df[df["Set"] == s].index[-1] - df[df["Set"] == s].index[0]
    df.loc[df["Set"] == s, "Duration"] = duration.seconds

duration_df = df.groupby(["category"])["Duration"].mean()

duration_df.iloc[0]

# Butterworth lowpass filter
df_lowpass = df.copy()
LowPass = LowPassFilter()
cutoff = 1
fs = 1000 / 200
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["Set"] == 2]

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5
)

for col in predict_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# Principal component analysis PCA
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(df_pca, predict_columns)
plt.figure(figsize=(20, 10))
plt.plot(range(1, len(pca_values) + 1), pca_values, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.show()

df_pca = PCA.apply_pca(df_pca, predict_columns, 3)
subset = df_pca[df_pca["Set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot(figsize=(20, 10))
plt.title("PCA components for set 35")

# Sum of squares attributes
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyro_r = (
    df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2
)

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

subset = df_squared[df_squared["Set"] == 35]
subset[["acc_r", "gyro_r"]].plot(figsize=(20, 10), subplots=True)

# Temporal abstraction
df_temporal = df_squared.copy()
NumericalAbstraction = NumericalAbstraction()
predicted_columns = predict_columns + ["acc_r", "gyro_r"]

ws = int(1000 / 200)
for col in predicted_columns:
    df_temporal = NumericalAbstraction.abstract_numerical(
        df_temporal, [col], ws, "mean"
    )
    df_temporal = NumericalAbstraction.abstract_numerical(df_temporal, [col], ws, "std")

df_temp_list = []
for s in df_temporal["Set"].unique():
    subset = df_temporal[df_temporal["Set"] == s].copy()
    for col in predicted_columns:
        subset = NumericalAbstraction.abstract_numerical(subset, [col], ws, "mean")
        subset = NumericalAbstraction.abstract_numerical(subset, [col], ws, "std")
    df_temp_list.append(subset)
df_temporal = pd.concat(df_temp_list)
df_temporal.info()

subset[["acc_x", "acc_x_temp_mean_ws_5", "acc_x_temp_std_ws_5"]].plot()
subset[["gyro_x", "gyro_x_temp_mean_ws_5", "gyro_x_temp_std_ws_5"]].plot()

# Frequency features
df_frequency = df_temporal.copy().reset_index()
FourierTransformation = FourierTransformation()

fs = 5
ws = int(2800 / 200)

df_frequency = FourierTransformation.abstract_frequency(df_frequency, ["acc_y"], ws, fs)
df_frequency.columns
subset = df_frequency[df_frequency["Set"] == 35]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_freq_0.0_Hz_ws_14",
        "acc_y_freq_0.714_Hz_ws_14",
        "acc_y_freq_1.429_Hz_ws_14",
    ]
].plot()
subset.columns
df_freq_list = []
for s in df_frequency["Set"].unique():
    print(f"Processing set {s}")
    subset = df_frequency[df_frequency["Set"] == s].reset_index(drop=True).copy()
    subset = FourierTransformation.abstract_frequency(subset, predicted_columns, ws, fs)
    df_freq_list.append(subset)
df_frequency = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
df_frequency.info()

# Dealing with overlapping windows
df_frequency = df_frequency.dropna()
df_frequency = df_frequency.iloc[::2]

# Clustering
df_clustering = df_frequency.copy()
cluster_col = [
    "acc_x",
    "acc_y",
    "acc_z",
]
k_val = range(2, 10)
inertia = []

for k in k_val:
    subset = df_clustering[cluster_col]
    Kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_label = Kmeans.fit_predict(subset)
    inertia.append(Kmeans.inertia_)

plt.figure(figsize=(20, 10))
plt.plot(k_val, inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

Kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_clustering[cluster_col]
df_clustering["cluster"] = Kmeans.fit_predict(subset)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection="3d")
for c in df_clustering["cluster"].unique():
    subset = df_clustering[df_clustering["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("3D Scatter Plot of Clusters")
plt.legend()
plt.show()

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection="3d")
for label in df_clustering["label"].unique():
    subset = df_clustering[df_clustering["label"] == label]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"{label}")
ax.set_xlabel("acc_x")
ax.set_ylabel("acc_y")
ax.set_zlabel("acc_z")
ax.set_title("3D Scatter Plot of Clusters")
plt.legend()
plt.show()

# Export dataset
df_clustering.to_pickle("../../data/interim/03_features_extracted.pkl")
