import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

ALPHA_GRID = np.linspace(1e-3, 1e2, 300)
C_GRID = np.logspace(-3, 2, 60)

def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Split into train/test, then standardise using training statistics only."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    return [scaler.transform(X_train), scaler.transform(X_test), y_train, y_test]

def get_regression_data() -> list[np.ndarray]:
    data = load_diabetes()
    return preprocess(data.data, data.target)

def get_classification_data() -> list[np.ndarray]:
    data = load_breast_cancer()
    return preprocess(data.data, data.target)

def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit an unregularised least-squares model."""
    return LinearRegression().fit(X, y)

def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit a ridge (L2) model, choosing `alpha` from ALPHA_GRID."""
    grid = GridSearchCV(Ridge(), param_grid={'alpha': ALPHA_GRID})
    return grid.fit(X, y).best_estimator_

def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit a lasso (L1) model, choosing `alpha` from ALPHA_GRID."""
    grid = GridSearchCV(Lasso(), param_grid={'alpha': ALPHA_GRID})
    return grid.fit(X, y).best_estimator_

def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit logistic regression with no regularisation at all."""
    # за умовою, вимикаємо штраф через C=np.inf
    return LogisticRegression(C=np.inf).fit(X, y)

def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit L2-penalised logistic regression, tuning `C` over C_GRID."""
    #  за інструкцією для sklearn >= 1.8, L2 - це l1_ratio=0
    model = LogisticRegression(l1_ratio=0, penalty='elasticnet', solver='saga', max_iter=5000) # solver='saga', бо універсально підтримує l1_ratio
    grid = GridSearchCV(model, param_grid={'C': C_GRID})
    return grid.fit(X, y).best_estimator_

def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit L1-penalised logistic regression, tuning `C` over C_GRID."""
    # L1 це l1_ratio=1. lbfgs не підтримує L1, тому використовуємо saga
    model = LogisticRegression(l1_ratio=1, penalty='elasticnet', solver='saga', max_iter=5000)
    grid = GridSearchCV(model, param_grid={'C': C_GRID})
    return grid.fit(X, y).best_estimator_
