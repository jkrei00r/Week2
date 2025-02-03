#hw2c.py
from NumericalMethods import GaussSeidel

def main():
    # Define the augmented matrices
    Aaug1 = [
        [3, 1, -1, 2],  # 3x + y - z = 2
        [1, 4, 1, 12],  # x + 4y + z = 12
        [2, 1, 2, 10]   # 2x + y + 2z = 10
    ]
    Aaug2 = [
        [1, -10, 2, 4],  # x - 10y + 2z = 4
        [3, 1, 4, 2],    # 3x + y + 4z = 2
        [9, 2, 3, 0]     # 9x + 2y + 3z = 0
    ]

    # Initial guesses
    x1 = [0, 0, 0]  # Initial guess for system 1
    x2 = [0, 0, 0]  # Initial guess for system 2

    # Solve the systems
    solution1 = GaussSeidel(Aaug1, x1, Niter=15)  # Solve system 1
    solution2 = GaussSeidel(Aaug2, x2, Niter=15)  # Solve system 2

    # Print results
    print("Solution for system 1:", solution1)
    print("Solution for system 2:", solution2)


if __name__ == "__main__":
    main()