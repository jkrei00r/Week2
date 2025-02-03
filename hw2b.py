#hw2b.py
from NumericalMethods import Secant
import math

def main():
    # Define the functions
    f1 = lambda x: x**2 - 2  # f(x) = x^2 - 2
    f2 = lambda x: x**2 - 2  # f(x) = x^2 - 2
    f3 = lambda x: math.exp(-x) - x  # f(x) = e^(-x) - x

    # Solve the equations
    root1 = Secant(f1, 1, 2, maxiter=5, xtol=1e-4)  # First equation
    root2 = Secant(f2, 1, 2, maxiter=15, xtol=1e-8)  # Second equation
    root3 = Secant(f3, 1, 2, maxiter=3, xtol=1e-8)  # Third equation

    # Print results
    print(f"Root of x^2 - 2 with maxiter=5, xtol=1e-4: {root1:.6f}")
    print(f"Root of x^2 - 2 with maxiter=15, xtol=1e-8: {root2:.6f}")
    print(f"Root of e^(-x) - x with maxiter=3, xtol=1e-8: {root3:.6f}")


if __name__ == "__main__":
    main()