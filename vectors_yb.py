 
from typing import Sequence

import numpy as np
from scipy import sparse

def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim."""
    # Ensure it's explicitly a column vector (dim, 1)
    return np.random.rand(dim, 1)

def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim."""
    return sparse.random(dim, 1, density=0.1, format='coo')

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition."""
    return x + y

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar."""
    return x * a

def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors."""
    result = np.zeros_like(vectors[0], dtype=float)
    for v, c in zip(vectors, coeffs):
        result += v * c
    return result

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product."""
    return float(np.sum(x * y))

def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max."""
    return float(np.linalg.norm(x, ord=order))

def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors."""
    return float(np.linalg.norm(x - y))

def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Angle between two vectors, in degrees."""
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
        
    cos_theta = np.sum(x * y) / (norm_x * norm_y)
    
    # FLOATING POINT FIX: Clip value to [-1, 1] to prevent NaN in arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cos_theta)))

def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal."""
    # FLOATING POINT FIX: Use isclose instead of strict == 0.0
    return bool(np.isclose(np.sum(x * y), 0.0, atol=1e-8))

def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations."""
    return np.linalg.solve(a, b)
