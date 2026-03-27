def main():
    """Main function to demonstrate basic programming concepts."""
    print("=== Concept Review ===\n")
    print("Sum of 5 + 3 = ", add(5, 3))
    print("Factorial of 20 = ", factorial(20))


def add(a, b):
    """Adds two numbers and returns the result."""
    return a + b

def factorial(n):
    """Calculates the factorial of a number."""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

if __name__ == "__main__":
    main()
