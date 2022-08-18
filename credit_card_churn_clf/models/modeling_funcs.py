from typing import Optional

import joblib
from loguru import logger
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score

from credit_card_churn_clf.data.data_funcs import read_credit_card_churn_split
from credit_card_churn_clf.models.feature_engineering import col_transf
from credit_card_churn_clf.utils import export_pkl, add_logger


def get_model(model: str, random_state: int = 42):
    models_dict = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                random_state=random_state, solver="saga", class_weight="balanced"
            ),
            "params_grid": {
                "clf__penalty": ["l1", "l2", "elasticnet", "none"],
                "clf__C": stats.uniform(loc=0, scale=10),
                "clf__l1_ratio": stats.uniform(loc=0, scale=1),
            },
        },
        "DecisionTreeClassifier": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "params_grid": {
                "clf__min_samples_split": stats.beta(a=3, b=70),
                "clf__min_samples_leaf": stats.beta(a=3, b=250),
                "clf__class_weight": [None, "balanced"],
            },
        },
        "RandomForestClassifier": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "params_grid": {
                "clf__n_estimators": [100, 150, 200],
                "clf__min_samples_split": stats.beta(a=2, b=70),
                "clf__min_samples_leaf": stats.beta(a=2, b=250),
                "clf__max_depth": [None, 5, 10, 15, 20],
                "clf__max_features": stats.beta(a=200, b=80),
                "clf__class_weight": [None, "balanced", "balanced_subsample"],
            },
        },
        "GradientBoostingClassifier": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "params_grid": {
                "clf__loss": ["log_loss", "exponential"],
                "clf__learning_rate": stats.uniform(loc=0, scale=1),
                "clf__subsample": stats.uniform(loc=0, scale=1),
                "clf__n_estimators": [50, 100, 150],
                "clf__min_samples_split": stats.beta(a=3, b=70),
                "clf__min_samples_leaf": stats.beta(a=3, b=250),
                "clf__max_depth": [None, 5, 10],
                "clf__max_features": stats.uniform(loc=0, scale=1),
            },
        },
    }
    return models_dict[model]


@add_logger(type="training")
@export_pkl
def train_model(
    data: str,
    model: str,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_iter: int = 50,
    random_state: int = 42,
    export_pkl_to: Optional[str] = None,
    print_log: bool = False,
):
    X_train, y_train = read_credit_card_churn_split(
        path=data, train_or_test="train", split_y=True
    )

    model_ = get_model(model, random_state=random_state)

    model_pipe = Pipeline([("col_transf", col_transf), ("clf", model_["estimator"])])

    model_cv = RandomizedSearchCV(
        model_pipe,
        model_["params_grid"],
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring,
        n_iter=n_iter,
    )

    model_cv.fit(X_train, y_train)
    return model_cv


def get_best_model(models_path: list[str]):
    best_models = [0] * len(models_path)
    model_names = [0] * len(models_path)
    best_scores = [0] * len(models_path)
    for i, m in enumerate(models_path):
        model = joblib.load(m)
        best_model = model.best_estimator_
        model_name = best_model.named_steps["clf"].__class__.__name__
        best_score = model.best_score_
        logger.info(
            f"""O melhor modelo de {model_name} tem score no conjunto de validação
            de {best_score}."""
        )
        best_models[i] = best_model
        model_names[i] = model_name
        best_scores[i] = best_score
    logger.info(
        f"O melhor modelo entre os avaliados é o {model_name[np.argmax(best_scores)]}."
    )
    return best_models[np.argmax(best_scores)]


@export_pkl
def predict_on_test_set(
    data: str,
    best_model: RandomizedSearchCV,
    threshold: float = 0.5,
    export_pkl_to: Optional[str] = None,
    print_log: bool = False,
):
    X_test, y_test = read_credit_card_churn_split(
        path=data, train_or_test="test", split_y=True
    )
    model_name = best_model.named_steps["clf"].__class__.__name__
    y_scores = best_model.predict_proba(X_test)[:, 1] >= threshold
    roc_auc_test = roc_auc_score(y_test, y_scores)
    recall_test = recall_score(y_test, y_scores)
    precision_test = precision_score(y_test, y_scores)
    logger.info(
        f"""Métricas do modelo {model_name} no conjunto de teste usando {threshold} como threshold:
            ROC AUC: {roc_auc_test}
            Recall: {recall_test}
            Precision: {precision_test}"""
    )
    X_test["y_true"] = y_test
    X_test["y_preds"] = y_scores
    return X_test
