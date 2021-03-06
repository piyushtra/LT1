# Python3 program for the above approach

# Function to find GCD of a & b
# using Euclid Lemma
def gcd(a, b):
    # Base Case
    if (b == 0):
        return a

    return gcd(b, a % b)


# Function to find the LCM of all
# elements in arr[]
def findlcm(arr, n):
    # Initialize result
    ans = arr[0]

    # Iterate arr[] to find LCM
    for i in range(1, n):
        ans = (((arr[i] * ans)) //
               (gcd(arr[i], ans)))

    # Return the final LCM
    return ans


# Function to find the sum of N
# fraction in reduced form
def addReduce(n, num, den):
    # To store the sum of all
    # final numerators
    final_numerator = 0

    # Find the LCM of all denominator
    final_denominator = findlcm(den, n)

    # Find the sum of all N
    # numerators & denominators
    for i in range(n):
        # Add each fraction one by one
        final_numerator = (final_numerator +
                           (num[i]) * (final_denominator //
                                       den[i]))
    print("final_numerator",str(final_numerator))
    print("final_denominator",str(final_denominator))
    # Find GCD of final numerator and
    # denominator
    GCD = gcd(final_numerator,
              final_denominator)

    # Convert into reduced form
    # by dividing from GCD
    final_numerator //= GCD
    final_denominator //= GCD

    # Print the final fraction
    print(final_numerator, "/",
          final_denominator)


# Driver Code

# Given N
N = 3

# Given Numerator
arr1 = [1, 2, 5]

# Given Denominator
arr2 = [2, 1, 6]

# Function call
addReduce(N, arr1, arr2)

# This code is contributed by code_hunt
