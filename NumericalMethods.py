# NumericalMethods.py

import math  # Import the math module for mathematical functions like sqrt, pi, and exp

def GaussSeidel(Aaug, x, Niter=15):
    """
    Purpose: Use the Gauss-Seidel method to estimate the solution to a set of N linear equations expressed in matrix form as Ax = b.

    Parameters:
    Aaug: Augmented matrix [A | b] with N rows and N+1 columns.
    x: Initial guess vector (array) for the solution.
    Niter: Number of iterations

    Returns:
    x: The final solution vector after Niter iterations.
    """
    N = len(Aaug)  # Number of equations
    for _ in range(Niter):  # Perform Niter iterations
        for i in range(N):  # Iterate over each equation
            sigma = 0
            for j in range(N):  # Compute the sum of A[i][j] * x[j]
                if j != i:
                    sigma += Aaug[i][j] * x[j]
            # Update x[i] using the Gauss-Seidel formula
            x[i] = (Aaug[i][-1] - sigma) / Aaug[i][i]
    return x

def GaussianPDF(x, mu, sigma):
    """
    Gaussian/Normal Probability Density Function (PDF).
    :param x: The value at which to evaluate the PDF.
    :param mu: The mean (μ) of the distribution.
    :param sigma: The standard deviation (σ) of the distribution.
    :return: The value of the Gaussian PDF at x.
    """
    # Calculate the Gaussian PDF using the formula:
    # (1 / (sigma * sqrt(2pi))) * ^(-0.5 * ((x - mu) / sigma)^2)
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to find the root of fcn(x) in the neighborhood of x0 and x1.
    :param fcn: The function for which we want to find the root.
    :param x0: First initial guess.
    :param x1: Second initial guess.
    :param maxiter: Maximum number of iterations.
    :param xtol: Tolerance for convergence (|xnewest - xprevious| < xtol).
    :return: The final estimate of the root.
    """
    for i in range(maxiter):
        # Calculate the new estimate using the Secant formula
        f_x0 = fcn(x0)
        f_x1 = fcn(x1)
        if f_x1 - f_x0 == 0:  # Avoid division by zero
            return x1
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        # Check for convergence
        if abs(x_new - x1) < xtol:
            return x_new

        # Update x0 and x1 for the next iteration
        x0, x1 = x1, x_new

    return x1  # Return the most recent estimate if maxiter is reached

def Simpson13(f, a, b, n=1000):
    """
    Numerical integration using Simpson's 1/3 rule.
    :param f: The function to integrate.
    :param a: The lower limit of integration.
    :param b: The upper limit of integration.
    :param n: The number of intervals (must be even).
    :return: The approximate integral of f from a to b.
    """
    # Check if the number of intervals (n) is even, as required by Simpson's 1/3 rule
    if n % 2 != 0:
        raise ValueError("Number of intervals (n) must be even for Simpson's 1/3 rule.")

    # Calculate the width of each interval
    h = (b - a) / n

    # Initialize the integral with the values of the function at the endpoints (a and b)
    integral = f(a) + f(b)

    # Loop through each interval to compute the integral
    for i in range(1, n):
        x = a + i * h  # Calculate the x-value for the current interval
        if i % 2 == 0:
            integral += 2 * f(x)  # Add 2 * f(x) for even intervals
        else:
            integral += 4 * f(x)  # Add 4 * f(x) for odd intervals

    # Multiply by h/3 to complete Simpson's 1/3 rule formula
    integral *= h / 3
    return integral


def Probability(PDF, args, c, GT=True):
    """
    Calculate the probability P(x < c) or P(x > c) for a Gaussian distribution.
    :param PDF: The Gaussian PDF function.
    :param args: A tuple containing μ (mean) and σ (standard deviation).
    :param c: The upper limit of integration.
    :param GT: If True, calculate P(x > c). If False, calculate P(x < c).
    :return: The calculated probability.
    """
    # Unpack the mean (mu) and standard deviation (sigma) from the args tuple
    mu, sigma = args

    # Set the lower limit of integration to mu - 5sigma (covers most of the distribution)
    a = mu - 5 * sigma

    # Set the upper limit of integration to c
    b = c

    # Define the integrand function, which is the Gaussian PDF
    def integrand(x):
        return PDF(x, mu, sigma)  # Evaluate the Gaussian PDF at x

    # Calculate the integral of the Gaussian PDF from a to b using Simpson's 1/3 rule
    probability = Simpson13(integrand, a, b)

    # If GT is True, calculate P(x > c) as 1 - P(x < c)
    if GT:
        return 1 - probability
    # Otherwise, return P(x < c) directly
    else:
        return probability


def main():
    """
    Main function to demonstrate the functionality of the program.
    """
    # Example 1: Calculate P(x < 105 | N(100, 12.5))
    mu1, sigma1 = 100, 12.5
    c1 = 105
    prob1 = Probability(GaussianPDF, (mu1, sigma1), c1, GT=False)
    print(f"P(x < {c1} | N({mu1}, {sigma1})) = {prob1:.6f}")

    # Example 2: Use the Secant method to find the root of f(x) = x^3 - 2x^2 - 5
    def fcn(x):
        return x**3 - 2*x**2 - 5

    root = Secant(fcn, x0=1, x1=2, maxiter=10, xtol=1e-5)
    print(f"Root of f(x) = x^3 - 2x^2 - 5: {root:.6f}")


if __name__ == "__main__":
    main()