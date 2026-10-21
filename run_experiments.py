import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.linear_models import RegressionInput, ClassificationInput, ClassificationMetrics
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models


def load_student_data():
    df = pd.read_csv("Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Performance Index"]).to_numpy(dtype=float)
    y = df["Performance Index"].to_numpy(dtype=float)
    return X, y


def load_german_data():
    df = pd.read_csv("german.csv", sep=";")
    y = df["Creditability"].to_numpy(dtype=float)
    X = df.drop(columns=["Creditability"]).to_numpy(dtype=float)
    return X, y


def run_regression():
    print("--- ЭТАП 1: ЛИНЕЙНАЯ РЕГРЕССИЯ (Student_Performance) ---")
    X, y = load_student_data()

    # Разделяем данные
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем свою модель (функция 1)
    data = RegressionInput(
        train_features=X_tr, train_target=y_tr,
        test_features=X_te, test_target=y_te,
        learning_rate=0.05, epochs=800,
    )
    res = train_linear_regression(data)

    # Считаем ошибку на тесте
    test_mse = float(np.mean((res.predictions - y_te) ** 2))

    print(f"Обучение завершено.")
    print(f"Ошибка MSE на тестовой выборке: {test_mse:.4f}")
    print(f"(Для сравнения, если всегда предсказывать среднее: {np.mean((y_te - y_tr.mean()) ** 2):.4f})")
    print()
    return test_mse


def run_classification():
    print("--- ЭТАП 2 и 3: КЛАССИФИКАЦИЯ (german) ---")
    X, y = load_german_data()

    # Разделяем данные
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    data = ClassificationInput(
        train_features=X_tr, train_target=y_tr,
        test_features=X_te, test_target=y_te,
        learning_rate=0.1, epochs=1000,
    )


    print("1. Своя логистическая регрессия:")
    res = train_logistic_classifier(data)
    own_metrics = ClassificationMetrics(
        roc_auc=roc_auc_score(y_te, res.probabilities),
        accuracy=accuracy_score(y_te, res.predictions),
        precision=precision_score(y_te, res.predictions),
        recall=recall_score(y_te, res.predictions),
        f1=f1_score(y_te, res.predictions)
    )
    print(
        f"   [Тест] ROC-AUC: {own_metrics.roc_auc:.4f} | Accuracy: {own_metrics.accuracy:.4f} | F1: {own_metrics.f1:.4f}")
    print()

    print("2. Готовые модели из sklearn:")
    baseline_res = compare_classical_models(data)

    print(
        f"   [Тест] Logistic Regression: ROC-AUC {baseline_res.logistic_regression.roc_auc:.4f} | Accuracy {baseline_res.logistic_regression.accuracy:.4f} | F1 {baseline_res.logistic_regression.f1:.4f}")
    print(
        f"   [Тест] Decision Tree:       ROC-AUC {baseline_res.decision_tree.roc_auc:.4f} | Accuracy {baseline_res.decision_tree.accuracy:.4f} | F1 {baseline_res.decision_tree.f1:.4f}")
    print(
        f"   [Тест] KNN (k=5):           ROC-AUC {baseline_res.knn.roc_auc:.4f} | Accuracy {baseline_res.knn.accuracy:.4f} | F1 {baseline_res.knn.f1:.4f}")
    print()

    unique, counts = np.unique(res.predictions, return_counts=True)
    print(f"Своя модель предсказала на тесте: {dict(zip(unique, counts))} (где 0 - плохой заемщик, 1 - хороший)")

    return own_metrics, baseline_res


if __name__ == "__main__":
    test_mse = run_regression()
    own_metrics, baseline_res = run_classification()

    results = {
        "linear_regression": {
            "test_mse": test_mse
        },
        "classification": {
            "own_logistic": {
                "roc_auc": own_metrics.roc_auc, "accuracy": own_metrics.accuracy,
                "precision": own_metrics.precision, "recall": own_metrics.recall, "f1": own_metrics.f1
            },
            "logistic_regression": {
                "roc_auc": baseline_res.logistic_regression.roc_auc,
                "accuracy": baseline_res.logistic_regression.accuracy,
                "precision": baseline_res.logistic_regression.precision,
                "recall": baseline_res.logistic_regression.recall, "f1": baseline_res.logistic_regression.f1
            },
            "decision_tree": {
                "roc_auc": baseline_res.decision_tree.roc_auc, "accuracy": baseline_res.decision_tree.accuracy,
                "precision": baseline_res.decision_tree.precision, "recall": baseline_res.decision_tree.recall,
                "f1": baseline_res.decision_tree.f1
            },
            "knn": {
                "roc_auc": baseline_res.knn.roc_auc, "accuracy": baseline_res.knn.accuracy,
                "precision": baseline_res.knn.precision, "recall": baseline_res.knn.recall, "f1": baseline_res.knn.f1
            }
        }
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nРезультаты сохранены в experiment_results.json")