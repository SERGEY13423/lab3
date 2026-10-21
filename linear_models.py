import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    # Стандартизация по статистикам train
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # Защита от деления на 0
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    n, d = X_train_s.shape
    w = np.zeros(d)
    b = 0.0
    lr = data.learning_rate
    losses = []

    # Цикл градиентного спуска
    for _ in range(data.epochs):
        pred = X_train_s @ w + b
        err = pred - y_train

        # Считаем и сохраняем MSE
        losses.append(float(np.mean(err ** 2)))

        # Градиенты
        gw = (2.0 / n) * (X_train_s.T @ err)
        gb = (2.0 / n) * err.sum()

        # Шаг спуска
        w -= lr * gw
        b -= lr * gb

    # Предсказание на тесте
    predictions = X_test_s @ w + b

    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=losses,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    # Стандартизация
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    n, d = X_train_s.shape
    w = np.zeros(d)
    b = 0.0
    lr = data.learning_rate
    losses = []

    # Градиентный спуск
    for _ in range(data.epochs):
        z = X_train_s @ w + b
        # Сигмоида (переводим в вероятности)
        p = 1.0 / (1.0 + np.exp(-z))

        # Защита от log(0), чтобы не было ошибок NaN
        p = np.clip(p, 1e-15, 1 - 1e-15)

        # Бинарная кросс-энтропия
        loss = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
        losses.append(float(loss))

        # Градиенты (для логистической регрессии они выглядят так же, как для MSE, но с вероятностями)
        err = p - y_train
        gw = (1.0 / n) * (X_train_s.T @ err)
        gb = (1.0 / n) * err.sum()

        # Шаг спуска
        w -= lr * gw
        b -= lr * gb

    # Предсказание на тесте
    z_test = X_test_s @ w + b
    probabilities = 1.0 / (1.0 + np.exp(-z_test))
    # Если вероятность > 0.5, ставим класс 1, иначе 0
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=losses,
        probabilities=probabilities,
        predictions=predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=float).ravel()

    # Для Logistic Regression и KNN нужна стандартизация
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    def get_metrics(y_true, y_pred, y_prob):
        return ClassificationMetrics(
            roc_auc=roc_auc_score(y_true, y_prob),
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred)
        )

    # 1. Готовая Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_metrics = get_metrics(y_test, lr.predict(X_test_s), lr.predict_proba(X_test_s)[:, 1])

    # 2. Decision Tree (деревьям масштабирование не нужно)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_metrics = get_metrics(y_test, dt.predict(X_test), dt.predict_proba(X_test)[:, 1])

    # 3. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    knn_metrics = get_metrics(y_test, knn.predict(X_test_s), knn.predict_proba(X_test_s)[:, 1])

    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )
