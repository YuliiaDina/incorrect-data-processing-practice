import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -x

def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    # np.flip обертає порядок елементів уздовж усіх осей матриці
    return np.flip(x)

def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation"""
    
    # (1) переведення кута в радіани та обчислення тригонометрії
    alpha_rad = np.radians(alpha_deg)
    c = np.cos(alpha_rad)
    s = np.sin(alpha_rad)
    
    # FLOATING POINT FIX: потрібно прибрати машинний нуль для ідеально прямих кутів
    if np.isclose(c, 0.0, atol=1e-15): c = 0.0
    if np.isclose(s, 0.0, atol=1e-15): s = 0.0
    if np.isclose(c, 1.0, atol=1e-15): c = 1.0
    if np.isclose(s, 1.0, atol=1e-15): s = 1.0
    if np.isclose(c, -1.0, atol=1e-15): c = -1.0
    if np.isclose(s, -1.0, atol=1e-15): s = -1.0

    # (2) сформування матриць базових перетворень (для 2D простору)
    S = np.array([
        [scale[0], 0],
        [0, scale[1]]
    ])
    
    Sh = np.array([
        [1, shear[0]],
        [shear[1], 1]
    ])
    
    R = np.array([
        [c, -s],
        [s, c]
    ])
    
    T = np.array([
        [translate[0]], 
        [translate[1]]
    ])
    
    # (3) застосування перетворень
    # ** Масштабування -> Скіс -> Обертання -> Зсув --в такому порядку
    return R @ Sh @ S @ x + T    #рівняння: x_new = R @ Sh @ S @ x + T
