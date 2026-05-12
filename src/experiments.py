import pandas as pd


def create_experiment_record(
    experiment_name,
    model,
    features_version,
    threshold,
    cv_mean,
    accuracy,
    precision,
    recall,
    profit,
    notes,
    depth=None,
    iterations=None,
    learning_rate=None,
):
    record = {
        "experiment_name": experiment_name,
        "model": model,
        "features_version": features_version,
        "threshold": threshold,
        "cv_mean": cv_mean,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "profit": profit,
        "notes": notes,
        "depth": depth,
        "iterations": iterations,
        "learning_rate": learning_rate,
    }
    return record


# Your simplified experiments list
experiments = [
    create_experiment_record(
        experiment_name="lr_base_t03",
        model="logistic_regression",
        features_version="base",
        threshold=0.3,
        cv_mean=0.6861,
        accuracy=0.6029,
        precision=0.4514,
        recall=0.9194,
        profit=354540,
        notes="simple baseline",
        depth=None,
        iterations=None,
        learning_rate=None,
    ),
    create_experiment_record(
        experiment_name="rf_base_t03",
        model="random_forest",
        features_version="base",
        threshold=0.3,
        cv_mean=0.7477,
        accuracy=0.5789,
        precision=0.4400,
        recall=0.9920,
        profit=479640,
        notes="strong recall",
        depth=15,
        iterations=None,  # <-- ADDED COMMA HERE
        learning_rate=None,
    ),
    create_experiment_record(
        experiment_name="cat_base_t03",
        model="catboost",
        features_version="base",
        threshold=0.3,
        cv_mean=0.7617,
        accuracy=0.5779,
        precision=0.4397,
        recall=0.9968,
        profit=488180,
        notes="best final",
        depth=8,
        iterations=60,
        learning_rate=0.03,
    ),
    create_experiment_record(
        experiment_name="cat_extra_feat_t03",
        model="catboost",
        features_version="extra_features",
        threshold=0.3,
        cv_mean=0.7616,
        accuracy=0.5784,
        precision=0.4399,
        recall=0.9961,
        profit=486960,
        notes="new features did not help",
        depth=8,
        iterations=60,
        learning_rate=0.03,
    ),
]


df = pd.DataFrame(experiments)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

df_sorted = df.sort_values("profit", ascending=False)
print(df.to_string(index=False))

print("\n📊 Experiments sorted by profit (best first):")
print(
    df_sorted[
        ["experiment_name", "model", "features_version", "profit", "cv_mean", "notes"]
    ]
)
df_sorted.to_csv("artifacts/experiment_log.csv", index=False)