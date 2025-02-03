# hw2a.py

# Import
from NumericalMethods import Probability, GaussianPDF

from NumericalMethods import GaussSeidel

def main():
    # Define the augmented matrix [A | b]
    Aaug = [
        [3, 1, -1, 2],
        [2, 4, 1, 12],
        [-1, 2, 5, 10]
    ]

    # Initial guess vector
    x = [0, 0, 0]

    # Number of iterations
    Niter = 15

    # Call the GaussSeidel function
    solution = GaussSeidel(Aaug, x, Niter)

    # Print the solution
    print("Solution vector (x):", solution)

if __name__ == "__main__":
    main()

def main():
    """
    Main function to calculate and print probabilities for given normal distributions.
    This function demonstrates how to calculate probabilities for specific conditions
    using the Gaussian (normal) distribution.
    """
    # P(x < 105 | N(100, 12.5))
    # Define the mean (mu1) and standard deviation (sigma1) for the first normal distribution
    mu1, sigma1 = 100, 12.5
    # Define the value (c1) for which we want to calculate the probability P(x < 105)
    c1 = 105
    # Calculate the probability using the Probability function from NumericalMethods
    # GaussianPDF is the probability density function for the normal distribution
    # (mu1, sigma1) are the parameters of the normal distribution
    # GT=False indicates we want P(x < c1), not P(x > c1)
    prob1 = Probability(GaussianPDF, (mu1, sigma1), c1, GT=False)
    # Print the result with formatted output
    print(f"P(x<{c1:.2f}|N({mu1},{sigma1}))={prob1:.2f}")

    # P(x > mu + 2sigma | N(100, 3))
    # Define the mean (mu2) and standard deviation (sigma2) for the second normal distribution
    mu2, sigma2 = 100, 3
    # Calculate the value for this distribution
    c2 = mu2 + 2 * sigma2
    # Calculating the probability using Probability function from NumericalMethods
    # GT=True indicates we want P(x > c2), not P(x < c2)
    prob2 = Probability(GaussianPDF, (mu2, sigma2), c2, GT=True)
    # Print the result with formatted output
    print(f"P(x>{c2:.2f}|N({mu2},{sigma2}))={prob2:.2f}")

from NumericalMethods import Secant
import math

def main():
    """
    Main function to solve the given problems using the Secant method.
    """
    # Define the function for which we want to find the root
    def fcn1(x):
        return x**3 - 2*x**2 - 5  # f(x) = x^3 - 2x^2 - 5

    # Problem 1: x0=1, x1=2, maxiter=5, xtol=1e-4
    root1 = Secant(fcn1, x0=1, x1=2, maxiter=5, xtol=1e-4)
    print(f"Root for Problem 1: {root1:.6f}")

    # Problem 2: x0=1, x1=2, maxiter=15, xtol=1e-8
    root2 = Secant(fcn1, x0=1, x1=2, maxiter=15, xtol=1e-8)
    print(f"Root for Problem 2: {root2:.6f}")

    # Problem 3: x0=1, x1=2, maxiter=3, xtol=1e-8
    root3 = Secant(fcn1, x0=1, x1=2, maxiter=3, xtol=1e-8)
    print(f"Root for Problem 3: {root3:.6f}")

if __name__ == "__main__":
    main()

# This ensures that the main() function is called only when the script is executed directly,
# not when it is imported as a module in another script.
if __name__ == "__main__":
    main()