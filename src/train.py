"""Train and evaluate a Logistic Regression model on the UCI Sonar dataset."""

from pathlib import Path

import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 1
MODEL_PATH = Path("models/sonar_logistic_regression.joblib")


def load_dataset():
    """Fetch the Sonar dataset and return feature and target dataframes."""
    dataset = fetch_ucirepo(id=151)
    return dataset.data.features, dataset.data.targets.iloc[:, 0]


def main():
    X, y = load_dataset()
    print(f"Dataset shape: {X.shape}")
    print("\nClass distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    print("\nBaseline accuracy: "
          f"{accuracy_score(y_test, baseline_predictions):.4f}")

    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    )
    print("\nMean cross-validation accuracy: "
          f"{cv_results['test_accuracy'].mean():.4f}")
    print("Mean cross-validation macro F1: "
          f"{cv_results['test_f1_macro'].mean():.4f}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=["M", "R"]))
    print("Confusion matrix (labels ordered as M, R):")
    print(confusion_matrix(y_test, predictions, labels=["M", "R"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
