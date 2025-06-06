import numpy as np

def calculate_epsilon(n: float, k: float) -> complex:
    """
    Calculate the complex permittivity from refractive index (n) and extinction coefficient (k).
    
    Args:
        n (float): Refractive index
        k (float): Extinction coefficient
        
    Returns:
        complex: Complex permittivity (ε = ε' + iε'')
    """
    epsilon_r = n**2 - k**2  # Real part
    epsilon_i = 2 * n * k    # Imaginary part
    return epsilon_r + 1j * epsilon_i 