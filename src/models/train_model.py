import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
# from LearningAlgorithms import ClassificationAlgorithms

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# Create a training and test set
df = pd.read_pickle("../../data/interim/03_features_extracted.pkl")
df_train = df.drop(columns=["participant", "category", "Set"])

X = df_train.drop("label", axis=1)
y = df_train["label"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Y_train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Y_test")
plt.legend()
plt.title("Distribution of classes in the training and test set")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Split feature subsets
basic_features = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
squared_features = [
    "acc_r",
    "gyro_r",
]
pca_features = [col for col in df_train.columns if "pca" in col]
time_features = [col for col in df_train.columns if "_temp_" in col]
frequency_features = [
    col for col in df_train.columns if ("_freq" in col) or ("_pse" in col)
]
cluster_features = ["cluster"]

print(f"Basic features: {len(basic_features)}")
print(f"Squared features: {len(squared_features)}")
print(f"PCA features: {len(pca_features)}")
print(f"Time features: {len(time_features)}")
print(f"Frequency features: {len(frequency_features)}")
print(f"Cluster features: {len(cluster_features)}")

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + squared_features + pca_features))
feature_set_3 = list(
    set(basic_features + squared_features + pca_features + time_features)
)
feature_set_4 = list(
    set(
        basic_features
        + squared_features
        + pca_features
        + time_features
        + frequency_features
        + cluster_features
    )
)
# Perform forward feature selection using simple decision tree
learner = ClassificationAlgorithms()
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, x_train, y_train
)
selected_features = [
    "acc_x_freq_0.0_Hz_ws_14",
    "Duration",
    "acc_y_freq_0.0_Hz_ws_14",
    "acc_z_temp_mean_ws_5",
    "acc_x_freq_2.5_Hz_ws_14",
    "acc_r_freq_weighted",
    "acc_z_freq_0.357_Hz_ws_14",
    "gyro_r_freq_2.143_Hz_ws_14",
    "gyro_y_freq_1.429_Hz_ws_14",
    "gyro_x_freq_0.0_Hz_ws_14",
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores, marker="o")
plt.xlabel("Number of features")
plt.ylabel("Cross-validated Accuracy")
plt.title("Forward feature selection")
plt.show()


# Grid search for best hyperparameters and model selection
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Featue set 1: Basic features",
    "Featue set 2: Basic + squared + PCA features",
    "Featue set 3: Basic + squared + PCA + time features",
    "Featue set 4: All features",
    "Selected features: Forward feature selection",
]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    # Selecting only features that exist in x_train
    features_to_use = [
        feat for feat in possible_feature_sets[i] if feat in x_train.columns
    ]
    selected_train_X = x_train[features_to_use].reset_index(drop=True)
    selected_test_X = x_test[features_to_use].reset_index(drop=True)
    # Reset indices to ensure alignment
    y_train_aligned = y_train.reset_index(drop=True)
    y_test_aligned = y_test.reset_index(drop=True)

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train_aligned,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test_aligned, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train_aligned, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test_aligned, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train_aligned, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test_aligned, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# Create a grouped bar plot to compare the results
score_df.sort_values(by="accuracy", ascending=False)
plt.figure(figsize=(12, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.title("Model performance for different feature sets")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0.75, 1)
plt.legend(loc="lower right")
plt.show()

# Select best model and evaluate results
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    selected_train_X, y_train_aligned, selected_test_X, gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# Select train and test data based on participant
participant_df = df.drop(columns=["category", "Set"])
x_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

x_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

x_train = x_train.drop("participant", axis=1)
x_test = x_test.drop("participant", axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Y_train")
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Y_test")
plt.legend()
plt.title("Distribution of classes in the training and test set")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Use best model again and evaluate results
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# Try a simpler model with the selected features
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    x_train[feature_set_4], y_train, x_test[feature_set_4], gridsearch=False
)

accuracy1 = accuracy_score(y_test, class_test_y)
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
