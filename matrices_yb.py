import numpy as np
import sympy as sp

def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m."""
    return np.random.rand(n, m)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition."""
    return x + y

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar."""
    return x * a

def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product."""

    return x @ y # @ - оператор, є стандартом для матричного множення в NumPy (те саме, що np.matmul)

def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`."""
    return np.eye(dim)

def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix."""
    return np.linalg.inv(x)

def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix."""
    return x.T

def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product."""
    return x * y    # Звичайне множення * в NumPy виконує поелементне множення

def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis."""
    # SymPy.Matrix.rref() повертає кортеж: (зведена матриця, індекси базисних стовпців). **найбезпечніший метод знайти базис, бо SymPy враховує похибки наближень
    _, pivot_cols = sp.Matrix(x).rref()
    return tuple(pivot_cols)

def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max."""
    return float(np.linalg.norm(x, ord=order))
