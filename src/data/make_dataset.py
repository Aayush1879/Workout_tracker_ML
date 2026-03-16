import pandas as pd
from glob import glob

# Read single CSV file
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyro = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# List all data in data/raw/MetaMotion
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# Extract features from filename
data_path = "../../data/raw/MetaMotion\\"
f = files[6]

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category

# Read all files
acc_dfs = []
gyro_dfs = []

acc_set = 1
gyro_set = 1
for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_dfs.append(df)
    elif "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_dfs.append(df)

# Concatenate all at once after loop completes
acc_dfs = pd.concat(acc_dfs, ignore_index=True)
gyro_dfs = pd.concat(gyro_dfs, ignore_index=True)

acc_dfs[acc_dfs["set"] == 1]

# Working with datetimes

acc_dfs.info()
acc_dfs["timestamp"] = pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_dfs.index = pd.to_datetime(acc_dfs["epoch (ms)"], unit="ms")
gyro_dfs.index = pd.to_datetime(gyro_dfs["epoch (ms)"], unit="ms")

del acc_dfs["epoch (ms)"]
del acc_dfs["elapsed (s)"]
del acc_dfs["time (01:00)"]

del gyro_dfs["epoch (ms)"]
del gyro_dfs["elapsed (s)"]
del gyro_dfs["time (01:00)"]

# function
files = glob("../../data/raw/MetaMotion/*.csv")


def read_and_process_file(f):
    acc_dfs = []
    gyro_dfs = []
    acc_set = 1
    gyro_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_dfs.append(df)
        elif "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_dfs.append(df)
    acc_dfs = pd.concat(acc_dfs, ignore_index=True)
    gyro_dfs = pd.concat(gyro_dfs, ignore_index=True)
    acc_dfs.index = pd.to_datetime(acc_dfs["epoch (ms)"], unit="ms")
    gyro_dfs.index = pd.to_datetime(gyro_dfs["epoch (ms)"], unit="ms")

    del acc_dfs["epoch (ms)"]
    del acc_dfs["elapsed (s)"]
    del acc_dfs["time (01:00)"]

    del gyro_dfs["epoch (ms)"]
    del gyro_dfs["elapsed (s)"]
    del gyro_dfs["time (01:00)"]

    return acc_dfs, gyro_dfs


# Merging datasets
data_merged = pd.concat([acc_dfs.iloc[:, :3], gyro_dfs], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "Set",
]


# Resample data (frequency conversion)

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "Set": "last",
}
data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled.info()
data_resampled["Set"] = data_resampled["Set"].astype(int)

# Export dataset
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
