import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_METADATA_PATH,
    DATASET_PATH,
    EXPERIMENT_NAME,
    EXPERIMENTS,
    MAX_DF,
    MAX_FEATURES,
    MIN_DF,
    NGRAM_RANGE,
    RANDOM_STATE,
    SUBLINEAR_TF,
    TEST_SIZE,
    MLFLOW_TRACKING_URI,
)
from src.preprocessing import build_vectorizer

def load_data():
    df = pd.read_csv(DATASET_PATH)
    original_count = len(df)

    required = {"review", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing)}")

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    if not set(df["sentiment"].unique()).issubset({"positive", "negative"}):
        raise ValueError("A coluna sentiment deve conter apenas positive/negative.")

    duplicate_count = int(df.duplicated(subset=["review"]).sum())
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

    print(f"Dataset original: {original_count:,} registros")
    print(f"Reviews duplicadas removidas: {duplicate_count:,}")
    print(f"Dataset preparado: {len(df):,} registros")
    print(df["sentiment"].value_counts())
    print(f"Valores ausentes: {df.isna().sum().sum()}")
    return df, original_count, duplicate_count


def select_best_result(results):
    if not results:
        raise ValueError("Nenhum resultado foi produzido.")
    return max(results, key=lambda item: item["f1"])

def log_confusion_matrix(y_true, y_pred, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "positive"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    path = run_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df, original_count, duplicate_count = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for exp in EXPERIMENTS:
        print(f"\n===== {exp['name']} =====")
        started = time.time()

        vectorizer = build_vectorizer(MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF, SUBLINEAR_TF)
        model = MLPClassifier(
            hidden_layer_sizes=exp["hidden_layer_sizes"],
            activation=exp["activation"],
            learning_rate_init=exp["learning_rate_init"],
            batch_size=exp["batch_size"],
            max_iter=exp["max_iter"],
            alpha=exp["alpha"],
            solver="adam",
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=3,
            random_state=RANDOM_STATE,
        )

        pipeline = Pipeline([("tfidf", vectorizer), ("mlp", model)])

        with mlflow.start_run(run_name=exp["name"]) as run:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, pos_label="positive"),
                "recall": recall_score(y_test, y_pred, pos_label="positive"),
                "f1": f1_score(y_test, y_pred, pos_label="positive"),
                "train_time_seconds": time.time() - started,
            }

            mlflow.log_params({
                "dataset_rows_original": original_count,
                "dataset_rows_prepared": len(df),
                "duplicate_reviews_removed": duplicate_count,
                "hidden_layers": str(exp["hidden_layer_sizes"]),
                "learning_rate": exp["learning_rate_init"],
                "activation": exp["activation"],
                "batch_size": exp["batch_size"],
                "max_iter": exp["max_iter"],
                "alpha": exp["alpha"],
                "solver": "adam",
                "tfidf_max_features": MAX_FEATURES,
                "tfidf_ngram_range": str(NGRAM_RANGE),
                "tfidf_min_df": MIN_DF,
                "tfidf_max_df": MAX_DF,
                "tfidf_sublinear_tf": SUBLINEAR_TF,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics(metrics)

            report_path = ARTIFACTS_DIR / f"{exp['name']}_classification_report.txt"
            report_path.write_text(classification_report(y_test, y_pred), encoding="utf-8")
            cm_path = log_confusion_matrix(y_test, y_pred, ARTIFACTS_DIR / exp["name"])

            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            model_info = mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                input_example=["This movie was excellent!"],
                skops_trusted_types=[
                    "src.preprocessing.clean_review",
                    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
                ],
            )

            print(" | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "train_time_seconds"))

            results.append({
                "run_id": run.info.run_id,
                "run_name": exp["name"],
                "model_uri": model_info.model_uri,
                **{k: metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
            })

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    best = select_best_result(results)
    best_uri = best["model_uri"]

    (ARTIFACTS_DIR / "best_model_uri.txt").write_text(best_uri, encoding="utf-8")
    BEST_MODEL_METADATA_PATH.write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    results_df.to_csv(ARTIFACTS_DIR / "experiment_results.csv", index=False)

    print("\n===== MELHOR MODELO =====")
    print(results_df.to_string(index=False))
    print(f"\nMelhor modelo: {best['run_name']} (F1={best['f1']:.4f})")
    print(f"Best model URI: {best_uri}")

if __name__ == "__main__":
    main()
