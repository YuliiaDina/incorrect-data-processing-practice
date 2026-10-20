import numpy as np
import scipy.linalg

def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform LU decomposition of a matrix."""
    # Повертає кортеж матриць (P, L, U) через SciPy
    return scipy.linalg.lu(x)

def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform QR decomposition of a matrix."""
    # Ортогональна Q та верхня трикутна R через NumPy
    return np.linalg.qr(x)

def determinant(x: np.ndarray) -> np.ndarray:
    """Calculate the determinant of a matrix."""
    # Приведення float до np.ndarray для збереження сигнатури
    return np.asarray(np.linalg.det(x))

def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvalues and right eigenvectors of a matrix."""
    # Повертає пару: (власні значення, власні вектори)
    return np.linalg.eig(x)

def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Singular Value Decomposition (SVD) of a matrix."""
    # Повертає (U, S, Vh). Матриця Vh вже транспонована. S це одновимірний масив
    return np.linalg.svd(x)
