"""ECSE343 Assignment 0."""
__author__ = 'Joseph Saliba 260975670.'
__copyright__ = 'Own work with base file provided by McGill.'

# import statements
import matplotlib.pyplot as plt   # plotting
import numpy as np                # all of numpy, for now...


def FloatSystem(beta, t, L, U):
    """
    ! Deliverable 1
    Function that returns a 1D-array of the possible base-10
    numbers corresponding to the float system.
    """
    # Gets all possible permutations of `d_1, d_2, ..., d_t-1`.
    permutations = np.array(
        np.meshgrid(
            *[list(range(0, beta))]*t
        )
    ).T.reshape(-1, t)

    # Forms a set out of the equivalent base-10 possible numbers.
    # The list comprehension uses the formula provided in the handout.
    s = {
        sum(
            [d/(beta**b) for b, d in enumerate(p)]  # sums all `d/(beta**b)`.
        ) * (beta**e)  # and multiplies the result by `(beta**e)`.
        for e in range(L, U+1) for p in permutations  # for all exp and perms.
        if p[0] != 0  # excludes the cases where `d_0 == 0`.
    }

    return np.concatenate((
        np.array([0.]),  # adds `0` to the return array
        np.array(list(s)),
        -1 * np.array(list(s))  # adds negative values to the return array
    ))


"""
! Deliverable 2

1.  `b[0:3]` is equivalent to `b[0:3, :]` as b is a 2D array
    and only one pair of `start:end` values is specified.
    As a result, the output is a (3, 5) array with the values from b
    in rows index 0 to 2 inclusive, while considering all
    the columns. Note that since step is not specified,
    it defaults to 1, hence why the output takes all
    elements in the range.
    Similarly, `b[:, 0:3]` outputs a (4, 3) array with values
    from b in columns index 0 to 2 inclusive, while considering
    all the rows. Also note that since step is not specified,
    it defaults to 1, hence why the output takes all
    elements in the range.

2.  `b[:, :] = b[1, :]` works because of numpy broadcasting,
    `b[1, :]` is automatically extended from (5,) to (4, 5)
    to match b, and so, `b[:, :] = b[1, :]` is actually doing
    `b[i, j] = b[1, j]` for all i and j in b. However,
    `b[:, :] = b[:, 1]` doesn't work as is, since `b[:, 1]`
    returns a numpy array of shape (4,), and numpy cannot
    broadcast an array from shape (4,) to (4, 5) to match b.
    Since however numpy can extend from (5,) to (4, 5), we can
    append `None` to the end of `b[:, 1]` to give it that one
    extra element, to make the broadcasting legal.

3.  `np.array([a[i%3] for i in range(5)])`.
    This assumes numpy is imported as np.

4.  a[2] returns the element in `a` at index 2. a[[2]] returns
    a numpy array of elements of 'a' with indices in the given
    array. Since the given array is only `[2]`, a[[2]] returns
    the element in `a` at index 2 in a np array of shape (1,).
    `a[:2, np.newaxis]` would return the transpose of `a[:2]`,
    as the indexing creates a newaxis. As such, the output would
    consider the elements at index 0 and 1 of `a` and transpose
    them to form a numpy array of shape (2, 1).

5.  a[a] might not always work as `a` might contain one or more
    values that are equal or larger than `len(a)`. As a result,
    doing a[a] would try to access values by indices out of
    bounds of the array, resulting in an error. However, doing
    `a[a % a.shape[0]]` resolve that problem, as now the values
    that were initially equal or larger than `len(a)` will be
    mapped to a range between `0` and `len(a) - 1`, due to how
    modulo arithmetic works. Due to that, we have guaranteed
    that all the indices calls are legal. 
"""


def SlowDotProduct(x, y):
    """
    ! Deliverable 3
    returns value of Dot product of 2 vectors of same shape.
    `SlowDotProduct(x, y)` only uses the Python library.
    """
    return sum([x_i * y_i for x_i, y_i in zip(x, y)])


def FastDotProduct(x, y):
    """
    ! Deliverable 3
    returns value of Dot product of 2 vectors of same shape.
    `FastDotProduct(x, y)` uses numpy.
    """
    return np.dot(x, y)


def CatchTheNaN(M):
    """
    ! Deliverable 4
    Returns a (2,) numpy.array with the indices of 
    `nan` of the arrays x and y, where `M == x @ y`.
    """
    return np.array([
        np.where(np.isnan(M[:, 0:1]))[0],
        np.where(np.isnan(M[0]))[0]
    ]).flatten()


def LogSumExpNaive(x):
    """
    ! Deliverable 5
    Returns the value of the LogSumExp function.
    This function uses the naive version, ie, the
    First formula used in the handout.
    """
    return np.log(sum([np.exp(x_i) for x_i in x]))


def LogSumExpRobust(x):
    """
    ! Deliverable 5
    Returns the value of the LogSumExp function.
    This function uses the robust version, ie, the
    Second formula used in the handout.

    The reason this implementation of Log Sum is numerically
    robust, is because the exponential function e^(x) scales
    exponentially with x: By applying the robust version, we
    can see that the values passed to `e()` are reduced by m.
    And so by applying the trick, the function e() has lower
    chances to yield overly large numbers which will overflow
    the computation and yield numerical errors.
    """
    m = max(x)
    return m + np.log(sum([np.exp(x_i - m) for x_i in x]))

# ! End of A0.


if __name__ == "__main__":
    # Here are some example test routines for the deliverables.
    # Feel free to write and include your own tests here.
    # None of the code in this main block will count for any credit, i.e.,
    # it won't be graded but you need always follow the collaboration and plagiarism policies.

    # [Deliverable 1 example test]
    print("\n\n[Deliverable 1]\n")
    tmp = FloatSystem(2, 2, -1, 2)

    # Here's a concrete solution for a (2,2,-1,2) floating-point system to test against
    test_solution = np.array([0.,  0.5, -0.5,  1., -1.,  2., -2.,
                             4., -4., 0.75, -0.75,  1.5, -1.5,  3., -3.,  6., -6.])
    # sorting our solution to simplify comparisons
    test_solution = np.sort(test_solution)

    print("Does your (2,2,-1,2) example system match our test solution? " +
          str(np.allclose(np.sort(tmp), test_solution)))

    # Example visualization code: plot an asterisk at each perfectly representable value along the real line
    plt.title(
        'Perfectly representable numbers for the $(2,2,-1,2)$ floating point system')
    plt.plot(tmp, np.zeros(tmp.shape[0]), '*')
    plt.yticks([])
    plt.show()  # this is a blocking call; kill the plotting window to continue execution

    # [Deliverable 2 example tests]
    print("\n\n[Deliverable 2]\n")
    print(" Not applicable (text answers, only); however, you may want to use a breakpoint() here to tinker with code to help answer these questions...")

    # [Deliverable 3 example tests]
    print("\n\n[Deliverable 3]\n")
    x = np.random.rand(100)
    y = np.random.rand(100)

    # Test correctness...
    slowResult = SlowDotProduct(x, y)
    fastResult = FastDotProduct(x, y)
    print("Matching result: " + str(np.isclose(slowResult, fastResult)))

    # Test performance...
    # Yes, you can import any modules (including "illegal" ones) in __main__ for testing purposes!
    import timeit
    print("Running SlowDotProduct 10000 times takes " + str(timeit.timeit(
        "SlowDotProduct(x, y)", number=10000, globals=globals())) + " seconds.")
    print("Running FastDotProduct 10000 times takes " + str(timeit.timeit(
        "FastDotProduct(x, y)", number=10000, globals=globals())) + " seconds.")
    print("Running SlowDotProduct 10000 times with different random inputs takes " + str(timeit.timeit(
        "SlowDotProduct(np.random.rand(1000), np.random.rand(1000))", number=10000, globals=globals())) + " seconds.")
    print("Running FastDotProduct 10000 times with different random inputs takes " + str(timeit.timeit(
        "FastDotProduct(np.random.rand(1000), np.random.rand(1000))", number=10000, globals=globals())) + " seconds.")

    # [Deliverable 4 example test]
    print("\n\n[Deliverable 4]\n")
    N = 30
    x = np.random.rand(N)
    y = np.random.rand(N)
    NaN_idx = np.array([np.random.randint(N-1), np.random.randint(N-1)])
    x[NaN_idx[0]] = np.float64("nan")
    y[NaN_idx[1]] = np.float64("nan")

    # Create our matrix M of shape (N,N) as the outer product of x and y
    # Note: we could've used, e.g., np.outer or np.einsum, here;
    # instead, we'll show off some broadcasting and numpy's built-in product operator @
    M = x[:, np.newaxis] @ y[np.newaxis, :]
    print("Did you find the NaN? " + str(np.allclose(NaN_idx, CatchTheNaN(M))))

    # [Deliverable 5 example test]
    print("\n\n[Deliverable 5]\n")
    # seed a random floating point vector of shape (N,) with values between a fraction of the min and max signed integer range
    N = 1000
    z = np.random.rand(N) * (np.iinfo(np.int16).max/100 -
                             np.iinfo(np.int16).min/100) + np.iinfo(np.int16).min/100

    naive = LogSumExpNaive(z)
    robust = LogSumExpRobust(z)
    print("Do the results match within floating point precision? " +
          str(np.isclose(naive, robust)))
